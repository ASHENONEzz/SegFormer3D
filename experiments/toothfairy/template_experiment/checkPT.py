import torch
import numpy as np

def analyze_ct_volume(pt_file, intensity_threshold=0.01, label_filter=True):
    """
    三维CBCT数据分析函数
    参数：
        pt_file: .pt文件路径
        intensity_threshold: 有效体素强度阈值（默认0.01）
        label_filter: 是否过滤背景标签（默认True）
    """
    try:
        # 加载数据
        data = torch.load(pt_file)
        
        # 验证数据结构
        required_keys = ['volume', 'label', 'affine', 'spacing']
        if not all(k in data for k in required_keys):
            raise ValueError("数据文件缺少必要字段，请检查预处理流程")

        # 提取数据
        volume = data['volume'].numpy().squeeze(0)  # [D=160, H=320, W=320]
        label = data['label'].numpy().squeeze(0)    # [D=160, H=320, W=320]
        
        # 生成坐标网格
        z, y, x = np.indices(volume.shape)
        
        # 构建过滤条件
        mask = (volume > intensity_threshold)
        if label_filter:
            mask &= (label != 0)  # 过滤背景标签

        # 应用过滤
        valid_coords = np.column_stack([z[mask], y[mask], x[mask]])
        valid_volume = volume[mask]
        valid_labels = label[mask]

        # 统计信息
        print(f"===== 三维数据报告 =====")
        print(f"文件路径：{pt_file}")
        print(f"数据形状：{volume.shape}")
        print(f"原始体素总量：{volume.size:,}")
        print(f"有效体素数量：{valid_coords.shape[0]:,} ({valid_coords.shape[0]/volume.size:.1%})")
        print(f"空间分辨率：{data['spacing']} mm")
        print(f"标签类别分布：")
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        for lbl, cnt in zip(unique_labels, counts):
            print(f"  类别 {lbl:>2d}: {cnt:>8,} 体素")

        # 示例展示前20个体素
        print("\n示例体素（前20个有效点）：")
        print(f"{'坐标 (Z,Y,X)':<20} | {'CT值':<8} | 标签")
        print("-" * 45)
        for i in range(min(20, len(valid_coords))):
            coord = valid_coords[i]
            ct_val = valid_volume[i]
            lbl = valid_labels[i]
            print(f"{str(tuple(coord)):<20} | {ct_val:.4f}  | {lbl}")

    except Exception as e:
        print(f"分析失败：{str(e)}")

# 使用示例
if __name__ == "__main__":
    pt_path = r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\ToothFairy_Training_Data\ToothFairy2F_001\ToothFairy2F_001_modalities.pt"
    analyze_ct_volume(
        pt_path,
        intensity_threshold=0.1,  # 根据数据特性调整
        label_filter=True
    )