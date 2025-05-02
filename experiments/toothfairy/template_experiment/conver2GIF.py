import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import gc

# 读取.nii.gz文件
file_path = r'D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\toothfairy_raw_data\train\labelsTr\ToothFairy2F_005.nii.gz'
img = nib.load(file_path)

# 获取图像数据
img_data = img.get_fdata()

# 可以查看图像的形状
print("Image shape:", img_data.shape)

# 可以查看图像的头信息
print("Image header:", img.header)

# 保存图像数据
# np.save(r'D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\ToothFairy2F_001.npy', img_data.astype(np.float32))

# 显示图像的一个切片
# plt.imshow(img_data[:, :, img_data.shape[2]//2], cmap='gray')
# plt.title('Slice at the middle of the volume')
# plt.savefig(r'D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\image_ToothFairy2F_005.png')

# # 显示图像的所有切片成 gif
# # Function to create a GIF from 3D volume
# def create_gif(data, output_path='output.gif', fps=10):
#     """
#     优化后的GIF生成函数
    
#     参数：
#     - data: 三维numpy数组 [H, W, D]
#     - output_path: 输出路径
#     - fps: 帧率（建议5-15）
#     """
#     # 预处理：归一化数据
#     data = data.astype(np.float32)
#     data = (data - data.min()) / (data.max() - data.min()) * 255
    
#     # 创建图形对象
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.axis('off')
    
#     # 初始化图像显示
#     img = ax.imshow(data[:, :, 0], cmap='gray', animated=True)
    
#     # 正确的更新函数
#     def update(frame):
#         img.set_array(data[:, :, frame])
#         return [img]
    
#     # 生成动画
#     total_frames = data.shape[2]
#     ani = FuncAnimation(fig, 
#                        update, 
#                        frames=total_frames,
#                        interval=1000//fps,  # 正确计算间隔
#                        blit=True)
    
#     # 保存时显示进度
#     print("正在生成GIF...")
#     ani.save(output_path, 
#              writer='pillow', 
#              fps=fps,
#              progress_callback=lambda i, n: print(f'\r进度: {i+1}/{n}', end=''))
    
#     # 主动释放内存
#     plt.close(fig)
#     del ani
#     gc.collect()

# # 使用示例
# file_path = r'D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\toothfairy_raw_data\train\imageTr\ToothFairy2F_001.nii.gz'
# img_data = nib.load(file_path).get_fdata()

# # 方向校正（根据实际数据可能需要调整）
# # img_data = np.rot90(img_data, k=1, axes=(0, 1))
# # img_data = np.transpose(img_data, (1, 0, 2))
# img_data = img_data.transpose(2, 0, 1)  # 转置为 [D, H, W]
# create_gif(img_data, 
#           output_path=r'D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\ToothFairy2F_001.gif',
#           fps=12)  # 推荐12帧/秒