import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载 .pt 文件
volume = torch.load(r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\ToothFairy_Training_Data\ToothFairy2F_001\ToothFairy2F_001_modalities.pt", map_location="cpu")
label = torch.load(r"D:\Ashen\Desktop\CBCT_Project\SegFormer3D\data\toothfairy_seg\ToothFairy_Training_Data\ToothFairy2F_001\ToothFairy2F_001_label.pt", map_location="cpu")

# volume = volume["volume"]  # [1, D, H, W]
# label = label["label"]     # [1, D, H, W]

# 转为 numpy（去掉 batch/channel 维）
# volume = volume.squeeze(0).numpy()
# label = label.squeeze(0).numpy()
volume = volume.squeeze(0)
label = label.squeeze(0)

class MultiAxisViewer:
    def __init__(self, volume, label=None):
        self.volume = volume
        self.label = label
        self.slices = [
            volume.shape[0] // 2,
            volume.shape[1] // 2,
            volume.shape[2] // 2
        ]

        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.im_objs = []
        self.mask_objs = []

        for i, ax in enumerate(self.axes):
            img = self.get_slice(i)
            im = ax.imshow(img, cmap='gray')
            self.im_objs.append(im)

            if self.label is not None:
                mask = self.get_slice(i, label=True)
                mask_im = ax.imshow(mask, cmap='Reds', alpha=0.3)
                self.mask_objs.append(mask_im)
            else:
                self.mask_objs.append(None)

            ax.set_title(self.get_title(i))
        
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.tight_layout()
        plt.show()

    def get_slice(self, axis, label=False):
        data = self.label if label else self.volume
        if axis == 0:
            return data[self.slices[0], :, :]
        elif axis == 1:
            return data[:, self.slices[1], :]
        elif axis == 2:
            return data[:, :, self.slices[2]]

    def get_title(self, axis):
        axis_name = ['Z (Axial)', 'Y (Coronal)', 'X (Sagittal)'][axis]
        return f"{axis_name} - Slice {self.slices[axis] + 1}"

    def on_scroll(self, event):
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                direction = 1 if event.button == 'up' else -1
                max_index = self.volume.shape[i] - 1
                self.slices[i] = np.clip(self.slices[i] + direction, 0, max_index)
                
                # 更新图像
                self.im_objs[i].set_data(self.get_slice(i))
                if self.label is not None:
                    self.mask_objs[i].set_data(self.get_slice(i, label=True))
                ax.set_title(self.get_title(i))
                self.fig.canvas.draw_idle()
                break

# 启动查看器
MultiAxisViewer(volume, label)