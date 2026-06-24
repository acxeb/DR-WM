import math
import torch.nn as nn
import torch
from torchvision import transforms
import torch.nn.functional as F
from modules.ConvBlock import ConvBlock
from modules.ResBlock import ResBlock

class DW_Decoder(nn.Module):

    def __init__(self, bg_message_length, face_message_length, blocks=2, channels=64):
        super(DW_Decoder, self).__init__()

        self.bg_message_length = bg_message_length
        self.face_message_length = face_message_length

        self.bg_conv1 = ConvBlock(4, 16, blocks=blocks)
        self.bg_down1 = Down(16, 32, blocks=blocks)
        self.bg_down2 = Down(32, 64, blocks=blocks)
        self.bg_down3 = Down(64, 128, blocks=blocks)
        self.bg_down4 = Down(128, 256, blocks=blocks)

        self.bg_up3 = UP(256, 128)
        self.bg_att3 = ResBlock(128 * 2, 128, blocks=blocks)

        self.bg_up2 = UP(128, 64)
        self.bg_att2 = ResBlock(64 * 2, 64, blocks=blocks)

        self.bg_up1 = UP(64, 32)
        self.bg_att1 = ResBlock(32 * 2, 32, blocks=blocks)

        self.bg_up0 = UP(32, 16)
        self.bg_att0 = ResBlock(16 * 2, 16, blocks=blocks)

        self.bg_conv_1x1 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bg_message_layer = nn.Linear(bg_message_length * bg_message_length, bg_message_length)

        self.face_conv1 = ConvBlock(4, 16, blocks=blocks)
        self.face_down1 = Down(16, 32, blocks=blocks)
        self.face_down2 = Down(32, 64, blocks=blocks)
        self.face_down3 = Down(64, 128, blocks=blocks)
        self.face_down4 = Down(128, 256, blocks=blocks)

        self.face_up3 = UP(256, 128)
        self.face_att3 = ResBlock(128 * 2, 128, blocks=blocks)

        self.face_up2 = UP(128, 64)
        self.face_att2 = ResBlock(64 * 2, 64, blocks=blocks)

        self.face_up1 = UP(64, 32)
        self.face_att1 = ResBlock(32 * 2, 32, blocks=blocks)

        self.face_up0 = UP(32, 16)
        self.face_att0 = ResBlock(16 * 2, 16, blocks=blocks)

        self.face_conv_1x1 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.face_message_layer = nn.Linear(face_message_length * face_message_length, face_message_length)

    def _forward_one_branch(
        self,
        x_in,
        conv1, down1, down2, down3, down4,
        up3, att3, up2, att2, up1, att1, up0, att0,
        conv_1x1,
        message_layer,
        message_length
    ):
        d0 = conv1(x_in)
        d1 = down1(d0)
        d2 = down2(d1)
        d3 = down3(d2)
        d4 = down4(d3)

        u3 = up3(d4)
        u3 = torch.cat((d3, u3), dim=1)
        u3 = att3(u3)

        u2 = up2(u3)
        u2 = torch.cat((d2, u2), dim=1)
        u2 = att2(u2)

        u1 = up1(u2)
        u1 = torch.cat((d1, u1), dim=1)
        u1 = att1(u1)

        u0 = up0(u1)
        u0 = torch.cat((d0, u0), dim=1)
        u0 = att0(u0)

        residual = conv_1x1(u0)

        message_map = F.interpolate(
            residual,
            size=(message_length, message_length),
            mode='nearest'
        )
        message = message_map.view(message_map.shape[0], -1)
        message = message_layer(message)

        return message

    def forward(self, x, face_mask):
        face_mask = face_mask.float()
        bg_mask = 1.0 - face_mask

        x_bg_in = torch.cat([x, bg_mask], dim=1)

        x_face_in = torch.cat([x, face_mask], dim=1)

        hat_bg_message = self._forward_one_branch(
            x_bg_in,
            self.bg_conv1, self.bg_down1, self.bg_down2, self.bg_down3, self.bg_down4,
            self.bg_up3, self.bg_att3, self.bg_up2, self.bg_att2,
            self.bg_up1, self.bg_att1, self.bg_up0, self.bg_att0,
            self.bg_conv_1x1,
            self.bg_message_layer,
            self.bg_message_length
        )

        hat_face_message = self._forward_one_branch(
            x_face_in,
            self.face_conv1, self.face_down1, self.face_down2, self.face_down3, self.face_down4,
            self.face_up3, self.face_att3, self.face_up2, self.face_att2,
            self.face_up1, self.face_att1, self.face_up0, self.face_att0,
            self.face_conv_1x1,
            self.face_message_layer,
            self.face_message_length
        )

        return hat_bg_message, hat_face_message

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

