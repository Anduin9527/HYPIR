import torch
import torch.distributed as dist
from torch.optim import Optimizer

from .muon_optim import muon_update


class MuonWithAdamWOptimizer(Optimizer):
    """
    混合Muon与AdamW优化器

    这个优化器允许对网络中的不同参数使用不同的优化策略：
    1. 对于隐藏层权重矩阵使用Muon优化
    2. 对于嵌入层、输出层和标量参数（如偏置）使用AdamW

    用法示例：
        # 区分不同类型的参数
        hidden_params = [p for n, p in model.named_parameters() if 'weight' in n and p.ndim >= 2]
        other_params = [p for n, p in model.named_parameters() if p not in hidden_params]

        # 创建优化器
        optimizer = MuonWithAdamWOptimizer(
            [
                {'params': hidden_params, 'use_muon': True, 'lr': 0.02, 'momentum': 0.95, 'weight_decay': 1e-4},
                {'params': other_params, 'use_muon': False, 'lr': 3e-4, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01}
            ]
        )

    参数:
        param_groups: 参数组列表，每个组必须包含'use_muon'标志以指示使用哪种优化器
    """

    def __init__(self, param_groups):
        # 检查每个参数组是否有use_muon标志
        for group in param_groups:
            assert "use_muon" in group, "每个参数组必须包含'use_muon'标志"

            if group["use_muon"]:
                # Muon默认参数
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
            else:
                # AdamW默认参数
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.01)

        super(MuonWithAdamWOptimizer, self).__init__(param_groups, dict())

        # 检查是否处于分布式环境
        self.is_distributed = False
        try:
            self.is_distributed = dist.is_available() and dist.is_initialized()
        except:
            self.is_distributed = False

    def _adam_update(self, grad, exp_avg, exp_avg_sq, step, betas, eps):
        """AdamW更新计算"""
        exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
        exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

        # 偏差校正
        bias_correction1 = 1 - betas[0] ** step
        bias_correction2 = 1 - betas[1] ** step

        # 计算更新
        denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
        update = exp_avg / bias_correction1 / denom

        return update

    @torch.no_grad()
    def step(self, closure=None):
        """执行单个优化步骤"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                # Muon优化
                lr = group["lr"]
                momentum_beta = group["momentum"]
                weight_decay = group["weight_decay"]
                ns_steps = group["ns_steps"]
                nesterov = group["nesterov"]

                if not self.is_distributed:
                    # 单设备情况
                    for p in group["params"]:
                        if p.grad is None:
                            continue

                        # 应用权重衰减
                        if weight_decay != 0:
                            p.mul_(1 - lr * weight_decay)

                        # 获取状态并初始化
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        # 计算Muon更新
                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=momentum_beta,
                            ns_steps=ns_steps,
                            nesterov=nesterov,
                        )

                        # 应用更新
                        p.add_(update.reshape(p.shape), alpha=-lr)
                else:
                    # 分布式情况
                    params = group["params"]
                    world_size = dist.get_world_size()
                    params_pad = params + [
                        torch.empty_like(params[-1] if params else torch.tensor(0.0))
                    ] * ((world_size - len(params) % world_size) % world_size)

                    for base_i in range(0, len(params), world_size):
                        rank = dist.get_rank()
                        if base_i + rank < len(params):
                            p = params[base_i + rank]
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)

                            # 应用权重衰减
                            if weight_decay != 0:
                                p.mul_(1 - lr * weight_decay)

                            # 获取状态并初始化
                            state = self.state[p]
                            if len(state) == 0:
                                state["momentum_buffer"] = torch.zeros_like(p)

                            # 计算Muon更新
                            update = muon_update(
                                p.grad,
                                state["momentum_buffer"],
                                beta=momentum_beta,
                                ns_steps=ns_steps,
                                nesterov=nesterov,
                            )

                            # 应用更新
                            p.add_(update.reshape(p.shape), alpha=-lr)

                        # 在所有进程间同步参数
                        if len(params) > 0:
                            if base_i + rank < len(params_pad):
                                send_tensor = params_pad[base_i + rank]
                            else:
                                send_tensor = params_pad[-1]
                            dist.all_gather(
                                params_pad[base_i : base_i + world_size], send_tensor
                            )
            else:
                # AdamW优化
                lr = group["lr"]
                betas = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad

                    # 应用权重衰减 (AdamW风格)
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    # 获取状态并初始化
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    # 更新step计数
                    state["step"] += 1

                    # 计算AdamW更新
                    update = self._adam_update(
                        grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        betas,
                        eps,
                    )

                    # 应用更新
                    p.add_(update, alpha=-lr)

        return loss
