import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from ..metrics.registry import BACKBONE

# 确保先导入 efficientnetb4
from .efficientnetb4 import EfficientNetB4
from .xception import Xception

# 添加调试信息
print("Registered backbones:", BACKBONE.data.keys())
