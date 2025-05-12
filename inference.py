import os
import argparse
import pandas as pd

from utils import FolderDataset
from utils import Runner
import torchvision.transforms as Transforms
import albumentations as A
from albumentations import ImageOnlyTransform
import cv2
import numpy as np
from PIL import Image
import yaml
import torch.backends.cudnn as cudnn
import torch
import random
from utils.detectors.ucf_detector import UCFDetector
from utils.detectors.spsl_detector import SpslDetector
from utils.detectors.efficientnetb4_detector import EfficientDetector
from utils.networks import efficientmodel
from utils.networks.efficientmodel import Detector

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def get_opts():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--your-team-name",
        type=str,
        default='安全对抗小组',
    )
    arg.add_argument(
        "--data-folder",
        type=str,
        default='./utils/data/test1'
    )
    arg.add_argument(
        "--model_weights",
        type=str,
        default='/home/huangjingjing/Security/detection_update/utils/sbi_ckpt_best.tar'
        # default='./utils/efficientnetb4_ckpt_best.pth'
    )
    arg.add_argument(
        "--result-path",
        type=str,
        default='./result'
    )

    arg.add_argument('--detector_path', type=str,
                        default='./utils/efficientnetb4.yaml',
                        help='path to detector YAML file')
    arg.add_argument("--test_dataset", nargs="+")

    opts = arg.parse_args()
    return opts


def get_dataset(opts, config):
    ### tips: customize your transforms
    class IsotropicResize(ImageOnlyTransform):  # 继承自 ImageOnlyTransform
        def __init__(
            self,
            max_side,
            interpolation_down=cv2.INTER_AREA,
            interpolation_up=cv2.INTER_CUBIC,
            always_apply=False,  # 新增 Albumentations 必要参数
            p=1.0,               # 新增概率参数 p
        ):
            # 调用父类构造函数初始化 always_apply 和 p
            super(IsotropicResize, self).__init__(always_apply=always_apply, p=p)
            self.max_side = max_side
            self.interpolation_down = interpolation_down
            self.interpolation_up = interpolation_up

        def apply(self, img, **params):  # 重命名为 apply 方法（Albumentations 规范）
            h, w = img.shape[:2]

            # 计算缩放比例以保持长宽比
            if max(h, w) == h:
                ratio = self.max_side / h
            else:
                ratio = self.max_side / w

            # 根据缩放方向选择插值方法
            if ratio > 1:
                interpolation = self.interpolation_up
            else:
                interpolation = self.interpolation_down

            # 计算新的尺寸
            new_h = int(h * ratio)
            new_w = int(w * ratio)

            # 调整图像大小
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
            return img

    # 从配置中提取参数
    config_copy = config.copy()
    resolution = config_copy.get('resolution', 299)

    # 检查是否存在data_aug参数，若不存在则使用默认值
    data_aug = config_copy.get('data_aug', {})
    flip_prob = data_aug.get('flip_prob', 0.5)
    rotate_limit = data_aug.get('rotate_limit', 10)
    rotate_prob = data_aug.get('rotate_prob', 0.5)
    blur_limit = data_aug.get('blur_limit', 3)
    blur_prob = data_aug.get('blur_prob', 0.5)
    brightness_limit = data_aug.get('brightness_limit', 0.2)
    contrast_limit = data_aug.get('contrast_limit', 0.2)
    quality_lower = data_aug.get('quality_lower', 75)
    quality_upper = data_aug.get('quality_upper', 100)

    # 定义均值和标准差，从配置中提取或使用默认值
    mean = config_copy.get('mean', [0.5, 0.5, 0.5])
    std = config_copy.get('std', [0.5, 0.5, 0.5])

    # 检查是否使用数据增强
    use_data_augmentation = config_copy.get('use_data_augmentation', True)

    # 创建自定义的转换函数
    class CustomTransforms:
        def __init__(self, use_aug=True):
            self.use_aug = use_aug

            # 创建增强转换
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=flip_prob),
                A.Rotate(limit=rotate_limit, p=rotate_prob),
                A.GaussianBlur(blur_limit=blur_limit, p=blur_prob),
                A.OneOf([
                    IsotropicResize(max_side=resolution, interpolation_down=cv2.INTER_AREA,
                                    interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=resolution, interpolation_down=cv2.INTER_AREA,
                                    interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=resolution, interpolation_down=cv2.INTER_LINEAR,
                                    interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit),
                    A.FancyPCA(),
                    A.HueSaturationValue()
                ], p=0.5),
                A.ImageCompression(quality_lower=quality_lower, quality_upper=quality_upper, p=0.5)
            ])

            # 创建基本转换
            self.basic_transform = A.Compose([
                IsotropicResize(max_side=resolution, interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_CUBIC),
            ])

            # 定义标准化转换
            self.normalize = Transforms.Normalize(mean=mean, std=std)
            self.to_tensor = Transforms.ToTensor()

        def __call__(self, img):
            # 将PIL图像转换为numpy数组
            if isinstance(img, Image.Image):
                img = np.array(img)

            # 应用数据增强或基本转换
            if self.use_aug:
                img = self.aug_transform(image=img)['image']
            else:
                img = self.basic_transform(image=img)['image']

            # 转换回PIL图像
            img = Image.fromarray(img)

            # 应用ToTensor和Normalize
            img = self.to_tensor(img)
            img = self.normalize(img)

            return img

    # 根据配置创建转换
    transforms = CustomTransforms(use_aug=use_data_augmentation)

    transforms_test = Transforms.Compose(
        [
            Transforms.Resize((299, 299)),
            Transforms.ToTensor(),
            Transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]
    )
    transforms_2 = Transforms.Compose([
        Transforms.Resize((380, 380)),  # 调整大小为380x380
        Transforms.ToTensor()           # 将PIL图像转换为tensor并自动将通道顺序从HWC转为CHW
    ])

    # 使用FolderDataset
    
    return FolderDataset(opts.data_folder, transforms_2)


def get_model_runner(opts, config, dataset):
    ### tips: customize your model
    #model = UCFDetector(config)
    # model = SpslDetector(config)
    model = Detector()
    # model=model.to(device)
    cnn_sd=torch.load(opts.model_weights)["model"]
    model.load_state_dict(cnn_sd)

    # DO NOT change Runner
    runner = Runner(model, dataset)
    return runner


if __name__ == "__main__":

    opts = get_opts()
    # parse options and load config
    with open(opts.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./utils/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict'] = config['label_dict']

    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    dataset = get_dataset(opts, config)
    runner = get_model_runner(opts, config, dataset)
    results = runner.run()

    os.makedirs(opts.result_path, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"))
    prediction_frame = pd.DataFrame(
        data = {
            "img_names": results["predictions"].keys(),
            "predictions": results["predictions"].values(),
        }
    )
    time_frame = pd.DataFrame(
        data = {
            "Data Volume": [len(results["predictions"].keys())],
            "Time": [results["time"]],
        }
    )
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()