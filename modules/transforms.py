import torch
from PIL import Image
from torchvision import transforms

image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

normalize_img = transforms.Normalize(image_mean, image_std)
unnormalize_img = transforms.Normalize(-image_mean / image_std, 1 / image_std)
unstd_img = transforms.Normalize(0, 1 / image_std)
std_img = transforms.Normalize(0, image_std)

default_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_img,
])



def get_transforms(
        img_size: int,
):
    train_transform = transforms.Compose([  # 训练阶段的变换器
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return train_transform, val_transform
