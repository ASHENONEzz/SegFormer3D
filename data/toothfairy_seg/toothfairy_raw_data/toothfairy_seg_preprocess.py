import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from monai.data import MetaTensor
from monai.transforms import (
    Orientation,
    EnsureType,
)
from skimage.transform import resize

class CBCTPreprocessor:
    def __init__(
        self,
        root_dir: str,
        save_dir: str = "../ToothFairy_Training_Data",
        target_size: tuple = (160, 320, 320),  # Segformer3D输入尺寸
        crop_margin: int = 16                   # 动态裁剪边界
    ):
        """
        牙齿CBCT预处理管道
        root_dir: 原始数据目录结构
            ├── imageTr
            │   └── case1.nii.gz
            │   └── case2.nii.gz
            └── labelsTr
                └── case1.nii.gz
                └── case2.nii.gz
        """
        self.image_dir = os.path.join(root_dir, "imageTr")
        self.label_dir = os.path.join(root_dir, "labelsTr")
        self.save_dir = save_dir
        self.target_size = target_size
        self.crop_margin = crop_margin
        
        # 获取所有病例ID
        self.case_ids = [f.split('.')[0] for f in os.listdir(self.image_dir) 
                        if f.endswith(('.nii', '.nii.gz'))]

    def __len__(self):
        return len(self.case_ids)

    def _load_volume(self, case_id):
        """加载CBCT图像"""
        img_path = os.path.join(self.image_dir, f"{case_id}.nii.gz")
        img = nib.load(img_path)
        return img.get_fdata(), img.affine

    def _load_label(self, case_id):
        """加载分割标签"""
        label_path = os.path.join(self.label_dir, f"{case_id}.nii.gz")
        label = nib.load(label_path).get_fdata()
        return label.astype(np.uint8)

    def _ct_normalization(self, volume):
        """CBCT专用标准化（基于Hounsfield单位）"""
        # 牙齿CT典型值范围
        volume = np.clip(volume, -1000, 3000)  # 去除异常值
        volume = (volume + 1000) / 4000        # 归一化到[0,1]
        return volume

    def _dynamic_crop(self, volume):
        """自动裁剪有效区域"""
        # 创建mask定位有效区域
        mask = volume > 0.1
        coords = np.where(mask)
        min_z, max_z = np.min(coords[0]), np.max(coords[0])
        min_y, max_y = np.min(coords[1]), np.max(coords[1])
        min_x, max_x = np.min(coords[2]), np.max(coords[2])
        
        # 扩展边界
        min_z = max(0, min_z - self.crop_margin)
        max_z = min(volume.shape[0], max_z + self.crop_margin)
        min_y = max(0, min_y - self.crop_margin)
        max_y = min(volume.shape[1], max_y + self.crop_margin)
        min_x = max(0, min_x - self.crop_margin)
        max_x = min(volume.shape[2], max_x + self.crop_margin)
        
        return volume[min_z:max_z, min_y:max_y, min_x:max_x]

    def _resample_volume(self, volume, order=3):
        """各向同性重采样"""
        # 计算缩放因子
        # zoom_factor = [t/s for t, s in zip(self.target_size, volume.shape)]

        # 添加维度验证
        assert len(volume.shape) == 3, "输入必须为3D数组"

        # 使用样条插值
        resized = resize(
            volume, 
            (self.target_size[0], self.target_size[1], self.target_size[2]), # Z, Y, X
            order=order, 
            mode='constant', 
            cval=0, 
            anti_aliasing=True
        )
        # return resized
        return resized.astype(np.float32)  # 📍 添加数据类型转换

    def _process_single_case(self, case_id):
        """处理单个病例"""
        try:
            # ================= 1. 数据加载 =================
            # 加载原始CBCT图像和标签
            volume, affine = self._load_volume(case_id)  # [H, W, D] (NIFTI默认顺序)
            label = self._load_label(case_id)            # [H, W, D]

            # ================= 2. 图像预处理 =================
            # CT图像标准化
            volume = self._ct_normalization(volume)      # [H, W, D]
            # 动态裁剪有效区域
            volume = self._dynamic_crop(volume)          # [H', W', D']
            label = self._dynamic_crop(label)            # [H', W', D']

            # ================= 3. 重采样处理 =================
            # 各向同性重采样到目标尺寸
            volume = self._resample_volume(volume)       # [H=160, W=320, D=320]
            label = self._resample_volume(label, order=0) # [H=160, W=320, D=320]

            # ================= 4. 维度顺序调整 =================
            # 调整到MONAI标准顺序 (D, H, W) -> (C=1, H, W, D)
            volume = np.transpose(volume, (0, 1, 2))     # [D=160, H=320, W=320]
            volume = volume[np.newaxis, ...]             # [C=1, D=160, H=320, W=320]
            
            label = np.transpose(label, (0, 1, 2))       # [D=160, H=320, W=320]

            # ================= 5. 方向标准化 =================
            # 转换为MetaTensor并调整方向
            volume_tensor = MetaTensor(volume, affine=affine)  # 添加通道维度
            volume_tensor = Orientation(axcodes="RAS")(volume_tensor)
            # 移除Meta信息并转换回numpy
            volume_np = EnsureType(data_type="numpy", track_meta=False)(volume_tensor)
            # volume_tensor = EnsureType()(volume_tensor).array.squeeze(0)

            # ================= 6. 最终验证 =================
            # 维度验证
            assert volume_np.shape == (1, 160, 320, 320), \
                f"图像尺寸错误: {volume_np.shape} 应 (1,160,320,320)"
            assert label.shape == (160, 320, 320), \
                f"标签尺寸错误: {label.shape} 应 (160,320,320)"
            
            # 数据类型验证
            assert volume_np.dtype == np.float32, "图像必须为float32"
            assert label.dtype == np.float32, "标签必须为float32"

            # ================= 7. 保存数据 =================
            # 保存处理结果
            save_path = os.path.join(self.save_dir, case_id)
            os.makedirs(save_path, exist_ok=True)
            
            torch.save({
                'volume': torch.FloatTensor(volume_tensor),  # [C=1, D=160, H=320, W=320]
                'label': torch.LongTensor(label),            # [D=160, H=320, W=320] 
                'affine': affine,                            # 原始空间信息
                'spacing': (1.0, 1.0, 1.0)                   # 各向同性spacing
            }, os.path.join(save_path, f"{case_id}.pt"))

        except Exception as e:
            print(f"Error processing {case_id}: {str(e)}")

    def run(self, num_workers=4):
        """启动并行处理"""
        os.makedirs(self.save_dir, exist_ok=True)
        Parallel(n_jobs=num_workers)(
            delayed(self._process_single_case)(case_id)
            for case_id in tqdm(self.case_ids, desc="Processing CBCT")
        )

if __name__ == "__main__":
    # 使用示例
    preprocessor = CBCTPreprocessor(
        root_dir=r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\toothfairy_raw_data\train",
        save_dir=r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\ToothFairy_Training_Data",
        target_size=(160, 320, 320)  # 匹配Segformer3D输入尺寸
    )
    preprocessor.run(num_workers=1)