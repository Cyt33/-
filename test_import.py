import sys
from pathlib import Path

# 强制把项目根目录加入 Python 搜索路径
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

print(f"✅ 项目根目录已加入路径: {project_root}")

# 测试导入 OpenVLA 核心模块
try:
    # 先导入基础 prismatic 包
    import prismatic
    print("✅ prismatic 包导入成功！")

    # 再导入 OpenVLA 模型
    from prismatic.models.vla import OpenVLAModel
    print("✅ OpenVLAModel 导入成功！")

except Exception as e:
    print(f"❌ 导入失败: {e}")
    print("当前 Python 搜索路径:")
    for p in sys.path:
        print(f"  - {p}")
    print("\n项目根目录下的文件夹:")
    for item in project_root.iterdir():
        if item.is_dir():
            print(f"  - {item.name}")