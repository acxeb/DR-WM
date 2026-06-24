import torch.nn as nn
import torch
from torchvision import transforms
import torch.nn.functional as F
from modules.ConvBlock import ConvBlock
from modules.ResBlock import ResBlock

class DW_Encoder(nn.Module):

    def __init__(
        self,
        bg_message_length,
        face_message_length,
        blocks=2,
        channels=64,
        attention=None
    ):
        super(DW_Encoder, self).__init__()

        self.bg_message_length = bg_message_length
        self.face_message_length = face_message_length

        self.bg_conv1 = ConvBlock(4, 16, blocks=blocks)
        self.bg_down1 = Down(16, 32, blocks=blocks)
        self.bg_down2 = Down(32, 64, blocks=blocks)
        self.bg_down3 = Down(64, 128, blocks=blocks)
        self.bg_down4 = Down(128, 256, blocks=blocks)

        self.bg_up3 = UP(256, 128)
        self.bg_up2 = UP(128, 64)
        self.bg_up1 = UP(64, 32)
        self.bg_up0 = UP(32, 16)

        self.bg_linear3 = nn.Linear(bg_message_length, bg_message_length * bg_message_length)
        self.bg_linear2 = nn.Linear(bg_message_length, bg_message_length * bg_message_length)
        self.bg_linear1 = nn.Linear(bg_message_length, bg_message_length * bg_message_length)
        self.bg_linear0 = nn.Linear(bg_message_length, bg_message_length * bg_message_length)

        self.bg_conv_msg3 = ConvBlock(1, channels, blocks=blocks)
        self.bg_conv_msg2 = ConvBlock(1, channels, blocks=blocks)
        self.bg_conv_msg1 = ConvBlock(1, channels, blocks=blocks)
        self.bg_conv_msg0 = ConvBlock(1, channels, blocks=blocks)

        self.bg_att3 = ResBlock(128 * 2 + channels + 1, 128, blocks=blocks)
        self.bg_att2 = ResBlock(64 * 2 + channels + 1, 64, blocks=blocks)
        self.bg_att1 = ResBlock(32 * 2 + channels + 1, 32, blocks=blocks)
        self.bg_att0 = ResBlock(16 * 2 + channels + 1, 16, blocks=blocks)

        self.bg_head = nn.Conv2d(16 + 3 + 1, 3, kernel_size=1, stride=1, padding=0)

        self.face_conv1 = ConvBlock(4, 16, blocks=blocks)
        self.face_down1 = Down(16, 32, blocks=blocks)
        self.face_down2 = Down(32, 64, blocks=blocks)
        self.face_down3 = Down(64, 128, blocks=blocks)
        self.face_down4 = Down(128, 256, blocks=blocks)

        self.face_up3 = UP(256, 128)
        self.face_up2 = UP(128, 64)
        self.face_up1 = UP(64, 32)
        self.face_up0 = UP(32, 16)

        self.face_linear3 = nn.Linear(face_message_length, face_message_length * face_message_length)
        self.face_linear2 = nn.Linear(face_message_length, face_message_length * face_message_length)
        self.face_linear1 = nn.Linear(face_message_length, face_message_length * face_message_length)
        self.face_linear0 = nn.Linear(face_message_length, face_message_length * face_message_length)

        self.face_conv_msg3 = ConvBlock(1, channels, blocks=blocks)
        self.face_conv_msg2 = ConvBlock(1, channels, blocks=blocks)
        self.face_conv_msg1 = ConvBlock(1, channels, blocks=blocks)
        self.face_conv_msg0 = ConvBlock(1, channels, blocks=blocks)

        self.face_att3 = ResBlock(128 * 2 + channels + 1, 128, blocks=blocks)
        self.face_att2 = ResBlock(64 * 2 + channels + 1, 64, blocks=blocks)
        self.face_att1 = ResBlock(32 * 2 + channels + 1, 32, blocks=blocks)
        self.face_att0 = ResBlock(16 * 2 + channels + 1, 16, blocks=blocks)

        self.face_head = nn.Conv2d(16 + 3 + 1, 3, kernel_size=1, stride=1, padding=0)

    def _expand_message(self, message, linear_layer, conv_layer, target_h, target_w, msg_len):
        expanded = linear_layer(message)
        expanded = expanded.view(-1, 1, msg_len, msg_len)
        expanded = F.interpolate(
            expanded,
            size=(target_h, target_w),
            mode='nearest'
        )
        expanded = conv_layer(expanded)
        return expanded

    def _resize_mask(self, mask, target_h, target_w):
        return F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def _forward_one_branch(
        self,
        x_rgb,               # [B, 3, H, W]，
        mask,                # [B, 1, H, W]
        message,             # [B, L]
        conv1, down1, down2, down3, down4,
        up3, up2, up1, up0,
        linear3, linear2, linear1, linear0,
        conv_msg3, conv_msg2, conv_msg1, conv_msg0,
        att3, att2, att1, att0,
        head,
        message_length
    ):
        # 分支输入：区域抑制后的图像 + 对应 mask
        x_in = torch.cat([x_rgb, mask], dim=1)  # [B, 4, H, W]

        # 编码
        d0 = conv1(x_in)
        d1 = down1(d0)
        d2 = down2(d1)
        d3 = down3(d2)
        d4 = down4(d3)

        # H/8
        u3 = up3(d4)
        msg3 = self._expand_message(message, linear3, conv_msg3, d3.shape[2], d3.shape[3], message_length)
        mask3 = self._resize_mask(mask, d3.shape[2], d3.shape[3])
        u3 = torch.cat((d3, u3, msg3, mask3), dim=1)
        u3 = att3(u3)

        # H/4
        u2 = up2(u3)
        msg2 = self._expand_message(message, linear2, conv_msg2, d2.shape[2], d2.shape[3], message_length)
        mask2 = self._resize_mask(mask, d2.shape[2], d2.shape[3])
        u2 = torch.cat((d2, u2, msg2, mask2), dim=1)
        u2 = att2(u2)

        # H/2
        u1 = up1(u2)
        msg1 = self._expand_message(message, linear1, conv_msg1, d1.shape[2], d1.shape[3], message_length)
        mask1 = self._resize_mask(mask, d1.shape[2], d1.shape[3])
        u1 = torch.cat((d1, u1, msg1, mask1), dim=1)
        u1 = att1(u1)

        # H
        u0 = up0(u1)
        msg0 = self._expand_message(message, linear0, conv_msg0, d0.shape[2], d0.shape[3], message_length)
        mask0 = self._resize_mask(mask, d0.shape[2], d0.shape[3])
        u0 = torch.cat((d0, u0, msg0, mask0), dim=1)
        u0 = att0(u0)

        delta = head(torch.cat((x_rgb, u0, mask), dim=1))
        return delta

    def forward(self, x, bg_message, face_message, face_mask):
        face_mask = face_mask.float()
        bg_mask = 1.0 - face_mask

        x_bg = x
        x_face = x

        delta_bg = self._forward_one_branch(
            x_bg, bg_mask, bg_message,
            self.bg_conv1, self.bg_down1, self.bg_down2, self.bg_down3, self.bg_down4,
            self.bg_up3, self.bg_up2, self.bg_up1, self.bg_up0,
            self.bg_linear3, self.bg_linear2, self.bg_linear1, self.bg_linear0,
            self.bg_conv_msg3, self.bg_conv_msg2, self.bg_conv_msg1, self.bg_conv_msg0,
            self.bg_att3, self.bg_att2, self.bg_att1, self.bg_att0,
            self.bg_head,
            self.bg_message_length
        )

        delta_face = self._forward_one_branch(
            x_face, face_mask, face_message,
            self.face_conv1, self.face_down1, self.face_down2, self.face_down3, self.face_down4,
            self.face_up3, self.face_up2, self.face_up1, self.face_up0,
            self.face_linear3, self.face_linear2, self.face_linear1, self.face_linear0,
            self.face_conv_msg3, self.face_conv_msg2, self.face_conv_msg1, self.face_conv_msg0,
            self.face_att3, self.face_att2, self.face_att1, self.face_att0,
            self.face_head,
            self.face_message_length
        )

        delta_bg = delta_bg * bg_mask
        delta_face = delta_face * face_mask

        return delta_bg, delta_face

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(Down, self).__init__()
        self.layer = torch.nn.Sequential(
            ConvBlock(in_channels, in_channels, stride=2),
            ConvBlock(in_channels, out_channels, blocks=blocks)
        )

    def forward(self, x):
        return self.layer(x)


class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

