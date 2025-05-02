import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='Split medical dataset into training and validation sets.')
    parser.add_argument('--data_root', type=str, required=True, default='./',
                        help='Root directory containing the train/ folder')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Output directory for CSV files')
    parser.add_argument('--image_subdir', type=str, default='train/imageTr',
                        help='Subdirectory containing image files (relative to data_root)')
    parser.add_argument('--val_ratio', type=float, default=1/7,
                        help='Validation set ratio (default: 1/7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # 构建完整路径
    image_dir = os.path.join(args.data_root, args.image_subdir)
    
    # 收集病例数据
    cases = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.nii.gz'):
            # 生成相对路径（相对于data_root）
            rel_path = os.path.join(args.image_subdir, filename)
            # 提取病例ID (e.g. ToothFairy2F_001)
            case_id = filename.split('.')[0]  
            cases.append({
                'data_path': rel_path,
                'case_name': case_id
            })

    if not cases:
        raise ValueError(f"No NIfTI files found in {image_dir}")

    # 分割数据集
    train_df, val_df = train_test_split(
        pd.DataFrame(cases),
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True
    )

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'validation.csv'), index=False)

    print(f"Dataset split complete: {len(train_df)} training, {len(val_df)} validation cases")
    print(f"CSV files saved to: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main()