import torch
import torch.distributed as dist
import math
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class MuonOptimizer(Optimizer):
    """
    BasicSR适配版 Muon优化器 - MomentUm Orthogonalized by Newton-schulz
    
    https://kellerjordan.github.io/posts/muon/
    
    Muon内部运行标准SGD动量，然后执行正交化后处理步骤，
    在该步骤中，每个2D参数的更新都被替换为最近的正交矩阵。
    为了高效的正交化，我们使用Newton-Schulz迭代，它具有可以在bfloat16下稳定运行于GPU上的优点。

    Muon应该只用于隐藏权重层。输入嵌入、最终输出层以及任何内部增益或偏差都应该使用标准方法（如AdamW）进行优化。
    隐藏卷积权重可以使用Muon进行训练，方法是将它们视为2D，然后折叠它们的最后3个维度。

    参数:
        params: 要优化的参数
        lr: 学习率，以每次更新的谱范数为单位
        weight_decay: AdamW风格的权重衰减
        momentum: 动量值。这里通常0.95就可以了
        ns_steps: Newton-Schulz迭代步数
        nesterov: 是否使用Nesterov动量
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, ns_steps=5, nesterov=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps, nesterov=nesterov)
        super(MuonOptimizer, self).__init__(params, defaults)
        
        # 检查是否处于分布式环境
        self.is_distributed = False
        try:
            self.is_distributed = dist.is_available() and dist.is_initialized()
        except:
            self.is_distributed = False

    @torch.no_grad()
    def step(self, closure=None):
        """执行单个优化步骤
        
        Args:
            closure (callable): 重新评估模型并返回损失的闭包
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 提取优化器参数
            lr = group['lr']
            momentum_beta = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']
            
            # 单设备情况
            if not self.is_distributed:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    # 应用权重衰减
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    
                    # 获取状态并初始化
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    
                    # 计算更新
                    update = muon_update(
                        p.grad, 
                        state["momentum_buffer"], 
                        beta=momentum_beta,
                        ns_steps=ns_steps, 
                        nesterov=nesterov
                    )
                    
                    # 应用更新
                    p.add_(update.reshape(p.shape), alpha=-lr)
            
            # 分布式情况
            else:
                params = group["params"]
                # 填充参数列表，使其长度为world_size的整数倍
                world_size = dist.get_world_size()
                params_pad = params + [torch.empty_like(params[-1])] * ((world_size - len(params) % world_size) % world_size)
                
                for base_i in range(0, len(params), world_size):
                    rank = dist.get_rank()
                    if base_i + rank < len(params):
                        p = params[base_i + rank]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)  # 强制同步
                        
                        # 应用权重衰减
                        if weight_decay != 0:
                            p.mul_(1 - lr * weight_decay)
                        
                        # 获取状态并初始化
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        
                        # 计算更新
                        update = muon_update(
                            p.grad, 
                            state["momentum_buffer"], 
                            beta=momentum_beta,
                            ns_steps=ns_steps, 
                            nesterov=nesterov
                        )
                        
                        # 应用更新
                        p.add_(update.reshape(p.shape), alpha=-lr)
                    
                    # 在所有进程间同步参数
                    dist.all_gather(params_pad[base_i:base_i + world_size], 
                                    params_pad[base_i + rank] if base_i + rank < len(params_pad) else params_pad[-1])

        return loss
