import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceErase(nn.Module):

    def __init__(self, fill_mode="zero", fill_value=0.0):
        super().__init__()
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.is_malicious = True
        self.requires_mask = True

    def forward(self, img: torch.Tensor, face_mask: torch.Tensor):
        if self.fill_mode == "zero":
            fill = torch.zeros_like(img)

        elif self.fill_mode == "one":
            fill = torch.ones_like(img)

        elif self.fill_mode == "constant":
            fill = torch.full_like(img, self.fill_value)

        elif self.fill_mode == "gray":
            fill = torch.zeros_like(img)

        elif self.fill_mode == "mean":
            mean_val = img.mean(dim=(2, 3), keepdim=True)   # [B, 3, 1, 1]
            fill = mean_val.expand_as(img)
        else:
            raise ValueError(f"Unsupported fill_mode: {self.fill_mode}")

        img_out = img * (1.0 - face_mask) + fill * face_mask
        img_out = img_out.clamp(-1, 1)

        return img_out
