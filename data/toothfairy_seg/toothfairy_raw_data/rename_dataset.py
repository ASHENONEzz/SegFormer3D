import os
import argparse

def rename_nifti_files(root_dir, dry_run=False):
    """
    递归重命名指定目录下的NIfTI文件
    :param root_dir: 需要处理的根目录
    :param dry_run: 模拟运行模式（不实际执行重命名）
    """
    renamed_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # 仅处理以_0000.nii.gz结尾的文件
            if filename.endswith("_0000.nii.gz"):
                # 构建旧文件路径
                old_path = os.path.join(root, filename)
                
                # 生成新文件名（移除_0000）
                new_filename = filename.replace("_0000.nii.gz", ".nii.gz")
                new_path = os.path.join(root, new_filename)
                
                # 安全检测：避免覆盖已存在的文件
                if os.path.exists(new_path):
                    print(f"⚠️  Conflict detected: {new_path} already exists. Skipping.")
                    continue
                
                # 执行重命名操作
                if not dry_run:
                    os.rename(old_path, new_path)
                    print(f"✅ Renamed: {filename} -> {new_filename}")
                else:
                    print(f"🚧 Simulation: {filename} -> {new_filename}")
                
                renamed_count += 1

    print(f"\nTotal files renamed: {renamed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量重命名医疗影像文件")
    parser.add_argument("--root_dir", required=True, 
                      help="包含需要重命名文件的根目录路径")
    parser.add_argument("--dry_run", action="store_true",
                      help="模拟运行模式（不实际修改文件系统）")
    
    args = parser.parse_args()
    
    # 验证目录存在性
    if not os.path.isdir(args.root_dir):
        raise ValueError(f"指定目录不存在: {args.root_dir}")
    
    rename_nifti_files(args.root_dir, args.dry_run)