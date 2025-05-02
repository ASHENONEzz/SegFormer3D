# import os
# import torch
# import pandas as pd
# from torch.utils.data import Dataset

# class ToothFairyDataset(Dataset):
#     """
#     CBCT三维分割数据集加载类
#     数据目录结构：
#     root_dir/
#     ├── train_fold_0.csv
#     ├── validation_fold_0.csv
#     └── ToothFairy_Training_Data/
#         ├── case_001/
#         │   └── case_001.pt
#         ├── case_002/
#         │   └── case_002.pt
#     pt文件结构:
#     {
#         'volume': Tensor[Channels, Depth, Height, Width], 
#         'label': Tensor[Depth, Height, Width],
#         'affine': np.ndarray (4x4矩阵)
#     }
#     """

#     def __init__(
#         self, 
#         root_dir: str, 
#         is_train: bool = True, 
#         transform=None, 
#         fold_id: int = None,
#         use_affine: bool = False
#     ):
#         """
#         root_dir: 预处理数据根目录（包含CSV文件和ToothFairy_Training_Data文件夹）
#         is_train: 训练/验证模式切换
#         transform: 数据增强变换组合
#         fold_id: 交叉验证折数ID
#         use_affine: 是否返回affine矩阵（用于后处理）
#         """
#         super().__init__()
        
#         # 解析CSV文件路径
#         if fold_id is not None:
#             csv_name = f"{'train' if is_train else 'validation'}_fold_{fold_id}.csv"
#         else:
#             csv_name = "train.csv" if is_train else "validation.csv"
            
#         csv_path = os.path.join(root_dir, csv_name)
#         assert os.path.exists(csv_path), f"CSV文件不存在: {csv_path}"
        
#         # 加载元数据
#         self.metadata = pd.read_csv(csv_path)
#         self.transform = transform
#         self.use_affine = use_affine
        
#         # 路径前缀处理
#         # self.processed_dir = os.path.join(root_dir, "ToothFairy_Training_Data")
#         # self.metadata["data_path"] = self.metadata["data_path"].apply(
#         #     lambda x: os.path.join(self.processed_dir, x)
#         # )
#         self.root_dir = root_dir  # 保存原始根目录
#         self.processed_dir = os.path.abspath(os.path.join(root_dir, "ToothFairy_Training_Data"))  # 获取绝对路径

#     def __len__(self):
#         return len(self.metadata)

#     def _load_case(self, data_path, case_name):
#         """加载单个病例数据"""
#         # case_path = os.path.join(data_path, f"{case_name}.pt")
#         # assert os.path.exists(case_path), f"数据文件不存在: {case_path}"
#         # 使用规范化的路径处理
#         case_dir = os.path.normpath(os.path.join(self.processed_dir, case_name))
#         case_path = os.path.join(case_dir, f"{case_name}.pt")
    
#         # 添加路径存在性断言
#         assert os.path.exists(case_path), f"数据文件不存在: {case_path}"
        
#         data = torch.load(case_path)
#         return {
#             "volume": data["volume"].float(),
#             "label": data["label"].long(),
#             "affine": data["affine"] if self.use_affine else None
#         }

#     def __getitem__(self, idx):
#         # 获取元数据
#         data_path = self.metadata.iloc[idx]["data_path"]
#         case_name = self.metadata.iloc[idx]["case_name"]
        
#         # 加载数据
#         case_data = self._load_case(data_path, case_name)
        
#         # 构建数据字典
#         sample = {
#             "image": case_data["volume"],  # [C, D, H, W]
#             "label": case_data["label"]    # [D, H, W]
#         }
        
#         # 添加affine信息（如果需要）
#         if self.use_affine:
#             sample["affine"] = case_data["affine"]
        
#         # 应用数据增强
#         # if self.transform:
#         #     sample = self.transform(sample)
            
#         return sample

#     @staticmethod
#     def collate_fn(batch):
#         """自定义批次处理函数"""
#         images = torch.stack([item["image"] for item in batch])
#         labels = torch.stack([item["label"] for item in batch])
        
#         # 处理affine矩阵
#         if "affine" in batch[0]:
#             affines = [item["affine"] for item in batch]
#             return {"image": images, "label": labels, "affine": affines}
        
#         return {"image": images, "label": labels}
import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class ToothFairyDataset(Dataset):
    """
        Brats2017 task 1 dataset is the segmentation corpus of the data. This dataset class performs dataloading
        on an already-preprocessed brats2021 data which has been resized, normalized and oriented in (Right, Anterior, Superior) format.
        The csv file associated with the data has two columns: [data_path, case_name]
    MRI_TYPE are ["Flair", "T1w", "T1gd", "T2w"] and segmentation label is store separately
    """

    def __init__(
        self, root_dir: str, is_train: bool = True, transform=None, fold_id: int = None
    ):
        """
        root_dir: path to (BraTS2021_Training_Data) folder
        is_train: whether or nor it is train or validation
        transform: composition of the pytorch transforms
        fold_id: fold index in kfold dataheld out
        """
        super().__init__()
        if fold_id is not None:
            csv_name = (
                f"train_fold_{fold_id}.csv"
                if is_train
                else f"validation_fold_{fold_id}.csv"
            )
            csv_fp = os.path.join(root_dir, csv_name)
            assert os.path.exists(csv_fp)
        else:
            csv_name = "train.csv" if is_train else "validation.csv"
            csv_fp = os.path.join(root_dir, csv_name)
            assert os.path.exists(csv_fp)

        self.csv = pd.read_csv(csv_fp)
        self.transform = transform

    def __len__(self):
        return self.csv.__len__()

    def __getitem__(self, idx):
        # data_path = self.csv["data_path"][idx]
        data_path = r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\ToothFairy_Training_Data"
        case_name = self.csv["case_name"][idx]
        data_path = os.path.join(data_path, case_name)
        # e.g, BRATS_001_modalities.pt
        # e.g, BRATS_001_label.pt
        volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")
        # load the preprocessed tensors
        volume = torch.load(volume_fp)
        label = torch.load(label_fp)
        data = {"image": torch.from_numpy(volume).float(), "label": torch.from_numpy(label).float()}

        if self.transform:
            data = self.transform(data)

        return data
