import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceErase(nn.Module):
    """
    人脸擦除增强：
    输入图像和面部掩码，输出去除人脸后的图像。

    输入:
        img:       [B, 3, H, W]，建议范围 [-1, 1]
        face_mask: [B, 1, H, W] 或 [B, H, W]，1 表示人脸区域

    输出:
        img_out:   [B, 3, H, W]
    """

    def __init__(self, fill_mode="zero", fill_value=0.0):
        super().__init__()
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.is_malicious = True   # 方便放入恶意增强池
        self.requires_mask = True

    def forward(self, img: torch.Tensor, face_mask: torch.Tensor):
        if self.fill_mode == "zero":
            fill = torch.zeros_like(img)

        elif self.fill_mode == "one":
            fill = torch.ones_like(img)

        elif self.fill_mode == "constant":
            fill = torch.full_like(img, self.fill_value)

        elif self.fill_mode == "gray":
            # 对 [-1, 1] 图像来说，0 对应中灰
            fill = torch.zeros_like(img)

        elif self.fill_mode == "mean":
            # 用每张图像的全局均值填充
            mean_val = img.mean(dim=(2, 3), keepdim=True)   # [B, 3, 1, 1]
            fill = mean_val.expand_as(img)
        else:
            raise ValueError(f"Unsupported fill_mode: {self.fill_mode}")

        img_out = img * (1.0 - face_mask) + fill * face_mask
        img_out = img_out.clamp(-1, 1)

        return img_out