from .identity import Identity
from .jpeg import JpegTest, JpegCompression
from .kornia_noises import *
from .crop import *
from .resize import Resize
from noise.SimSwap_main.for_swap import SimSwap
from noise.HISD.for_use import HiSD
from noise.stargan.main import StarGAN
from noise.Uniface.for_use_swap import Uniface_swap
from noise.GANimation.main import GANimation

name2aug = {
    'resize': Resize,
    'identity': Identity,
    'jpeg': JpegTest,
    'jpeg_diff': JpegCompression,
    'gaussian_blur': GaussianBlur,
    'MedianBlur': MedianBlur,
    'brightness': Brightness,
    'contrast': Contrast,
    'saturation': Saturation,
    'hue': Hue,
    'Dropout': Dropout,
    'simswap': SimSwap,
    'HiSD': HiSD,
    'GANimation': GANimation,
    'stargan': StarGAN,
    'Uniface_swap': Uniface_swap,
}
class Augmenter(nn.Module):
    """
    Augments the watermarked image.
    Outputs two versions: Benign Augmented and Malicious Augmented.
    """

    def __init__(
            self,
            augs: dict,
            augs_params: dict,
            **kwargs: dict
    ) -> None:
        super(Augmenter, self).__init__()

        all_augs, all_probs = self.parse_augmentations(
            augs=augs,
            augs_params=augs_params
        )

        self.augs_benign = []
        self.probs_benign = []

        self.augs_malicious = []
        self.probs_malicious = []

        for aug, prob in zip(all_augs, all_probs):
            is_malicious = getattr(aug, 'is_malicious', False)

            if is_malicious:
                self.augs_malicious.append(aug)
                self.probs_malicious.append(prob)
            else:
                self.augs_benign.append(aug)
                self.probs_benign.append(prob)

        self.probs_benign = self._normalize_probs(self.probs_benign)
        self.probs_malicious = self._normalize_probs(self.probs_malicious)

    def _normalize_probs(self, probs_list):
        if len(probs_list) == 0:
            return torch.tensor([])
        probs_tensor = torch.tensor(probs_list)
        return probs_tensor / probs_tensor.sum()

    def parse_augmentations(self, augs: dict, augs_params: dict):
        augmentations = []
        probs = []
        for aug_name in augs.keys():
            aug_prob = float(augs[aug_name])
            aug_params = augs_params[aug_name] if aug_name in augs_params else {}
            try:
                selected_aug = name2aug[aug_name](**aug_params)
            except KeyError:
                raise ValueError(f"Augmentation {aug_name} not found. Add it in name2aug.")
            augmentations.append(selected_aug)
            probs.append(aug_prob)
        return augmentations, probs

    def apply_augmentation(self, imgs_w, imgs, pool, probs, attrs=None):

        if len(pool) == 0:
            return imgs_w, "Identity"

        index = torch.multinomial(probs, 1).item()
        selected_aug = pool[index]

        if isinstance(selected_aug, nn.Module):
            selected_aug = selected_aug.to(imgs_w.device)

        h, w = imgs_w.shape[-2:]

        img_in = imgs_w.unsqueeze(0)

        if isinstance(selected_aug, StarGAN) and attrs is not None:

            attr_in = attrs.unsqueeze(0)
            img_out = selected_aug(img_in, attr_in)

        elif selected_aug.__class__.__name__ == 'Dropout' and imgs is not None:
            img_out = selected_aug(img_in, imgs)

        else:
            img_out = selected_aug(img_in)

        return img_out.squeeze(0), str(selected_aug)

    def forward(self, imgs_w: torch.Tensor, imgs: torch.Tensor = None, attrs=None):
        """
        Returns:
            imgs_benign: (B, C, H, W)
            imgs_malicious: (B, C, H, W)
        """
        batch_size = imgs_w.shape[0]

        forward_images = imgs_w.clone().detach()

        noised_benign = torch.zeros_like(forward_images)
        noised_malicious = torch.zeros_like(forward_images)

        log_benign = []
        log_malicious = []

        for i in range(batch_size):
            orig_img_i = imgs[i] if imgs is not None else None

            img_b, name_b = self.apply_augmentation(
                forward_images[i], orig_img_i, self.augs_benign, self.probs_benign, attrs=None
            )
            noised_benign[i] = img_b
            log_benign.append(name_b)

            img_m, name_m = self.apply_augmentation(
                forward_images[i], orig_img_i, self.augs_malicious, self.probs_malicious,
                attrs=attrs[i] if attrs is not None else None
            )
            noised_malicious[i] = img_m
            log_malicious.append(name_m)

        noised_benign = noised_benign.clamp(-1, 1)
        noised_malicious = noised_malicious.clamp(-1, 1)

        gap_benign = noised_benign - forward_images
        gap_malicious = noised_malicious - forward_images

        imgs_benign_out = imgs_w + gap_benign
        imgs_malicious_out = imgs_w + gap_malicious

        return imgs_benign_out, imgs_malicious_out

    def __repr__(self) -> str:
        b_names = [aug.__class__.__name__ for aug in self.augs_benign]
        m_names = [aug.__class__.__name__ for aug in self.augs_malicious]
        return f"Augmenter(\n  Benign Pool={b_names},\n  Malicious Pool={m_names}\n)"