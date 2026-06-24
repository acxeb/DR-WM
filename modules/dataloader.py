import os
import torch
from PIL import Image
from torch.utils.data import Dataset



class ImageMaskAttrDataset(Dataset):
    def __init__(self, image_dir, mask_dir,
                 image_transform=None, mask_transform=None):
        super(ImageMaskAttrDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.attr_path = "CelebAMask-HQ/celeba-hq/CelebAMask-HQ-attribute-anno.txt"
        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.img_transform = image_transform
        self.mask_transform = mask_transform

        self.attr_dict = self._parse_attributes(self.attr_path)

        image_files = {}
        for f in os.listdir(image_dir):
            if self._is_image(f):
                file_id = self._get_id(f)
                image_files[file_id] = f

        mask_files = {}
        for f in os.listdir(mask_dir):
            if self._is_image(f):
                file_id = self._get_id(f)
                mask_files[file_id] = f

        common_ids = sorted(list(set(image_files.keys()) & set(mask_files.keys()) & set(self.attr_dict.keys())))

        self.data_list = []
        for fid in common_ids:
            self.data_list.append({
                'img_path': os.path.join(image_dir, image_files[fid]),
                'mask_path': os.path.join(mask_dir, mask_files[fid]),
                'label': self.attr_dict[fid]
            })

    def _get_id(self, filename):
        name = os.path.splitext(filename)[0]
        name = name.split('_')[0]
        return int(name)

    def _parse_attributes(self, attr_path):
        attr_dict = {}
        with open(attr_path, 'r') as f:
            lines = f.readlines()

        all_attr_names = lines[1].split()
        attr2idx = {name: i for i, name in enumerate(all_attr_names)}
        target_indices = [attr2idx[attr] for attr in self.selected_attrs]

        for line in lines[2:]:
            split = line.split()
            file_id = self._get_id(split[0])
            values = split[1:]

            label = [1.0 if values[i] == '1' else 0.0 for i in target_indices]
            attr_dict[file_id] = torch.FloatTensor(label)

        return attr_dict

    def _is_image(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]

        img = Image.open(item['img_path']).convert('RGB')
        mask = Image.open(item['mask_path']).convert('L')
        label = item['label']

        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return img, mask, label