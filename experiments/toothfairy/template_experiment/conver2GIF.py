import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from matplotlib.colors import ListedColormap

def cbct_with_mask_to_gif(image_path, mask_path, output_path,
                          slice_idx='middle',
                          rotation_speed=2,
                          duration=200,
                          ct_center=400,
                          ct_width=2000,
                          mask_alpha=0.3,
                          mask_cmap='Reds'):
    """
    生成带分割掩码的CBCT旋转GIF
    
    参数：
    - image_path: CBCT图像路径(.nii.gz)
    - mask_path: 分割掩码路径(.nii.gz)
    - output_path: 输出GIF路径
    - slice_idx: 切片索引（默认中间层）
    - rotation_speed: 每帧旋转角度（建议1-5）
    - duration: 动画总时长（毫秒）
    - ct_center: CT窗位（牙齿常用400）
    - ct_width: CT窗宽（牙齿常用2000）
    - mask_alpha: 掩码透明度（0-1）
    - mask_cmap: 掩码颜色映射（推荐'Reds','Greens'等）
    """
    # 加载数据
    ct_img = nib.load(image_path)
    mask_img = nib.load(mask_path)
    
    ct_data = ct_img.get_fdata().astype(np.float32)
    mask_data = mask_img.get_fdata().astype(np.uint8)
    
    # 方向校正（牙齿CBCT常用设置）
    ct_data = np.rot90(ct_data, k=1, axes=(0,1))  # 顺时针旋转90度
    ct_data = np.transpose(ct_data, (1,0,2))       # 交换XY轴
    
    mask_data = np.rot90(mask_data, k=1, axes=(0,1))
    mask_data = np.transpose(mask_data, (1,0,2))
    
    # 数据验证
    assert ct_data.shape == mask_data.shape, "图像与掩码维度不匹配！"
    
    # 选择切片
    if slice_idx == 'middle':
        slice_idx = ct_data.shape[2] // 2
    ct_slice = ct_data[:, :, slice_idx].T
    mask_slice = mask_data[:, :, slice_idx].T

    # 窗宽窗位调整
    min_val = ct_center - ct_width/2
    max_val = ct_center + ct_width/2
    ct_slice = np.clip(ct_slice, min_val, max_val)
    ct_slice = (ct_slice - min_val) / (max_val - min_val)  # 归一化到0-1

    # 创建带透明度支持的掩码颜色映射
    mask_cmap = plt.get_cmap(mask_cmap)
    mask_cmap_colors = mask_cmap(np.arange(mask_cmap.N))
    mask_cmap_colors[:, -1] = mask_alpha  # 设置透明度
    transparent_cmap = ListedColormap(mask_cmap_colors)

    # 初始化画布
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    
    # 绘制CT图像（冠状面视图）
    ct_plot = ax.imshow(ct_slice, 
                       cmap='gray', 
                       extent=[0, ct_slice.shape[1], 0, ct_slice.shape[0], 0, 0],
                       origin='lower')
    
    # 叠加分割掩码（提升2个单位避免Z-fighting）
    mask_plot = ax.imshow(np.ma.masked_where(mask_slice==0, mask_slice), 
                         cmap=transparent_cmap, 
                         extent=[0, mask_slice.shape[1], 0, mask_slice.shape[0], 2, 2],
                         origin='lower')

    # 设置3D视图参数
    ax.view_init(elev=15, azim=0)  # 初始视角
    ax.dist = 8  # 观察距离

    # 动画更新函数
    def update(frame):
        ax.view_init(elev=15, azim=frame*rotation_speed)
        return [ct_plot, mask_plot]

    # 生成动画
    total_frames = int(360 / rotation_speed)
    ani = FuncAnimation(fig, update, frames=total_frames, blit=True)

    # 保存GIF
    ani.save(output_path, 
             writer='pillow', 
             fps=1000/duration,
             progress_callback=lambda i, n: print(f'\r生成进度: {i+1}/{n}', end=''))

    plt.close(fig)
    
    # 优化GIF循环
    with imageio.get_reader(output_path) as reader:
        frames = [np.array(frame) for frame in reader]
    imageio.mimsave(output_path, frames, duration=duration/1000, loop=0)

    print(f"\n可视化结果已保存至：{output_path}")

# 使用示例
if __name__ == "__main__":
    # 牙齿CBCT典型参数设置
    cbct_with_mask_to_gif(
        image_path=r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\toothfairy_raw_data\train\imageTr\ToothFairy2F_001.nii.gz",
        mask_path=r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\toothfairy_raw_data\train\labelsTr\ToothFairy2F_001.nii.gz",
        output_path=r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\teeth_animation.gif",
        ct_center=400,      # 适合牙齿的窗位
        ct_width=2000,      # 骨窗宽
        mask_alpha=0.3,      # 适中的透明度
        mask_cmap='summer',  # 黄绿色调
        rotation_speed=2,    # 中等旋转速度
        duration=3000        # 3秒完成一圈
    )