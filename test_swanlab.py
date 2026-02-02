#!/usr/bin/env python3
"""
SwanLab 连接测试脚本
测试SwanLab是否正确配置和连接
"""

import sys

def test_swanlab():
    print("=" * 50)
    print("  SwanLab 连接测试")
    print("=" * 50)
    print()
    
    # 测试1: 导入SwanLab
    print("1. 测试导入 swanlab...")
    try:
        import swanlab
        print("   ✓ swanlab 导入成功")
        print(f"   版本: {swanlab.__version__}")
    except ImportError as e:
        print(f"   ✗ swanlab 导入失败: {e}")
        print("   请运行: pip install swanlab")
        return False
    print()
    
    # 测试2: 检查登录状态
    print("2. 检查登录状态...")
    try:
        # 尝试获取登录信息
        swanlab.login(relogin=False)
        print("   ✓ 已登录 SwanLab")
    except Exception as e:
        print(f"   ✗ 未登录或登录失败: {e}")
        print("   请运行: swanlab login")
        return False
    print()
    
    # 测试3: 创建测试实验
    print("3. 测试创建实验...")
    try:
        run = swanlab.init(
            project="hypir-test",
            experiment_name="connection-test",
            config={"test": True},
            mode="cloud"  # 确保在线模式
        )
        print("   ✓ 实验创建成功")
        print(f"   项目: hypir-test")
        print(f"   实验: connection-test")
    except Exception as e:
        print(f"   ✗ 实验创建失败: {e}")
        return False
    print()
    
    # 测试4: 记录测试数据
    print("4. 测试记录数据...")
    try:
        for step in range(1, 6):
            swanlab.log({
                "loss": 1.0 / step,
                "accuracy": step * 0.1,
                "test_metric": step ** 2
            }, step=step)
        print("   ✓ 数据记录成功")
        print("   已记录 5 个步骤的数据")
    except Exception as e:
        print(f"   ✗ 数据记录失败: {e}")
        swanlab.finish()
        return False
    print()
    
    # 测试5: 关闭实验
    print("5. 测试关闭实验...")
    try:
        swanlab.finish()
        print("   ✓ 实验关闭成功")
    except Exception as e:
        print(f"   ✗ 实验关闭失败: {e}")
        return False
    print()
    
    return True

def main():
    success = test_swanlab()
    
    print("=" * 50)
    if success:
        print("✅ 所有测试通过！")
        print()
        print("SwanLab 已正确配置，可以开始训练。")
        print("访问 https://swanlab.cn 查看测试实验。")
        print()
        print("测试实验会显示在项目 'hypir-test' 中，")
        print("你可以在网页上删除这个测试实验。")
    else:
        print("❌ 测试失败")
        print()
        print("请根据上面的错误信息进行排查：")
        print("1. 确保已安装 swanlab: pip install swanlab")
        print("2. 确保已登录: swanlab login")
        print("3. 检查网络连接")
        print("4. 访问 https://swanlab.cn/settings 获取 API Key")
    print("=" * 50)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
