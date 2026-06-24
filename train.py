import os
import sys
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
import omegaconf
import kornia
from tqdm import tqdm
from modules.Encoder_U import DW_Encoder
from modules.Decoder_U import DW_Decoder
from modules.identity_maker import IdentityWatermark
from noise.augmenter import Augmenter
from modules.dataloader import ImageMaskAttrDataset
from modules.transforms import get_transforms
from net import WatermarkNet
from modules.utils import *
from modules.PatchDiscriminator import Patch_Discriminator
from modules.logger import MetricLogger, redirect_stdout_to_file
from pytorch_wavelets import DTCWTForward

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="lpips")

parser = argparse.ArgumentParser()

group = parser.add_argument_group('Experiments parameters')
group.add_argument("--mode", type=str, default="train", required=False)
group.add_argument("--train_batch_size", type=int, default=8, required=False)
group.add_argument("--val_batch_size", type=int, default=8, required=False)
group.add_argument("--train_epoch", type=int, default=80, required=False)
group.add_argument("--eval_freq", type=int, default=5, required=False)
group.add_argument("--saveimg_freq", type=int, default=10, required=False)
group.add_argument("--message_length", type=int, default=64, required=False)
group.add_argument("--image_size", type=int, default=256, required=False)
group.add_argument("--message_range", type=float, default=0.1, required=False)
group.add_argument("--train_dir", type=str, default="/root/autodl-tmp/celeba-hq/256/train_256", required=False)
group.add_argument("--val_dir", type=str, default="/root/autodl-tmp/celeba-hq/256/val_256", required=False)
group.add_argument("--train_mask_dir", type=str, default="/root/autodl-tmp/celeba-hq/256/mask_train_256",
                   required=False)  # 256
group.add_argument("--val_mask_dir", type=str, default="/root/autodl-tmp/celeba-hq/256/mask_val_256",
                   required=False)  # 256
group.add_argument("--output_dir", type=str, default="output/train/",
                   help="Output directory for logs and images (Default: /output)")
group.add_argument("--checkpoint_dir", type=str, default="checkpoint_dir")
group.add_argument("--augmenter_config", type=str, default="/root/autodl-tmp/config/augs.yaml",
                   help="Path to the augmenter config file")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(args):
    train_transform, val_transform = get_transforms(img_size=args.image_size)
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = ImageMaskAttrDataset(args.train_dir, args.train_mask_dir, image_transform=train_transform,
                                         mask_transform=mask_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataset = ImageMaskAttrDataset(args.val_dir, args.val_mask_dir, image_transform=val_transform,
                                       mask_transform=mask_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True)

    Discriminator = Patch_Discriminator().to(device)

    embedder = DW_Encoder(args.message_length, args.message_length).to(device)
    extractor = DW_Decoder(args.message_length, args.message_length).to(device)

    augmenter_cfg = omegaconf.OmegaConf.load(args.augmenter_config)
    augmenter = Augmenter(augmenter_cfg["augs"], augmenter_cfg["params"]).to(device)
    print("Augmenter:", augmenter)
    net = WatermarkNet(embedder=embedder, extractor=extractor, augmenter=augmenter)
    net = net.to(device)

    optimizer_emb = Adam(net.embedder.parameters(), lr=2e-4, betas=(0.5, 0.99))
    optimizer_ext = Adam(net.extractor.parameters(), lr=2e-4, betas=(0.5, 0.99))
    optimizer_dis = torch.optim.Adam(
        Discriminator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.99)
    )
    epoch = 1

    checkpoint_path = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        state_dict = checkpoint['watermark_net_state_dict']
        Discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=True)
        own_state = net.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                if param.size() == own_state[name].size():
                    own_state[name].copy_(param)
                else:
                    print(f"  - Skipping layer '{name}' due to shape mismatch: "
                          f"checkpoint shape {param.size()} vs current model shape {own_state[name].size()}")

        print("Custom weight loading complete.")

        # 优化器调度器权重加载
        optimizer_emb.load_state_dict(checkpoint['optimizer_emb_state_dict'])
        optimizer_ext.load_state_dict(checkpoint['optimizer_ext_state_dict'])
        optimizer_dis.load_state_dict(checkpoint['optimizer_dis_state_dict'])
        epoch = checkpoint['epoch'] + 1

    for epoch in range(epoch, args.train_epoch + 1):
        log_file = redirect_stdout_to_file(epoch, log_dir="logs/train")
        original_stdout = sys.stdout
        sys.stdout = log_file
        print(f"Epoch {epoch} started at {datetime.now()}")

        train_one_epoch(
            watermark_net=net,
            optimizer_emb=optimizer_emb,
            optimizer_ext=optimizer_ext,
            optimizer_dis=optimizer_dis,
            discriminator=Discriminator,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
        )

        # 验证
        if epoch % args.eval_freq == 0:
            print('validating')
            validate_model(
                watermark_net=net,
                val_loader=val_loader,
                epoch=epoch,
                device=device
            )

        # 记录和打印结果
        print(f"Epoch {epoch} ended at {datetime.now()}")
        sys.stdout = original_stdout
        log_file.close()

        save_checkpoint(
            watermark_net=net,
            discriminator=Discriminator,
            optimizer_emb=optimizer_emb,
            optimizer_ext=optimizer_ext,
            optimizer_dis=optimizer_dis,
            epoch=epoch,
            path=os.path.join(args.checkpoint_dir, f'latest_model.pth')
        )

        # 定期保存检查点
        if epoch % 10 == 0:
            save_checkpoint(
                watermark_net=net,
                discriminator=Discriminator,
                optimizer_emb=optimizer_emb,
                optimizer_ext=optimizer_ext,
                optimizer_dis=optimizer_dis,
                epoch=epoch,
                path=os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            )


def redirect_stdout_to_file(epoch=None, log_dir="logs", is_train=True):
    os.makedirs(log_dir, exist_ok=True)
    if is_train:
        log_path = os.path.join(log_dir, f"epoch_{epoch:03d}.txt")
    else:
        log_path = os.path.join(log_dir, f"{epoch}.txt")
    return open(log_path, "w")


def train_one_epoch(
        watermark_net: WatermarkNet,
        optimizer_emb: torch.optim.Optimizer,
        optimizer_ext: torch.optim.Optimizer,
        discriminator,
        optimizer_dis,
        train_loader: DataLoader,
        device: torch.device,
        epoch: int,
):
    watermark_net.train()
    discriminator.train()
    torch.autograd.set_detect_anomaly(False)

    header = f'Train - Epoch: [{epoch}/{args.train_epoch}]'
    metric_logger = MetricLogger(delimiter="  ")

    criterion_MSE = nn.MSELoss().to(device)
    criterion_LPIPS = lpips.LPIPS(net='vgg').to(device).eval()
    criterion_LPIPS.requires_grad_(False)
    criterion_freq = PerceptualDTCWTLoss()
    accumulation_steps = 1
    optimizer_emb.zero_grad(set_to_none=True)
    optimizer_ext.zero_grad(set_to_none=True)
    optimizer_dis.zero_grad(set_to_none=True)
    label_cover = 1.0
    label_encoded = - 1.0

    for it, (imgs, masks, attrs) in enumerate(metric_logger.log_every(train_loader, 10, header)):

        imgs = imgs.to(device)
        masks = masks.to(device)
        B = imgs.shape[0]
        with torch.no_grad():
            message_bg = (
                    (torch.randint(0, 2, (imgs.shape[0], args.message_length), device=device).float() * 2 - 1)
                    * args.message_range
            )
            # message_face = identity_maker(imgs)
            message_face = (
                    (torch.randint(0, 2, (imgs.shape[0], args.message_length), device=device).float() * 2 - 1)
                    * args.message_range
            )
            prior = prior_extractor(imgs)
        delta_bg, delta_face = watermark_net.embedder(imgs, message_bg, message_face, masks)
        imgs_w = (imgs + delta_bg + delta_face).clamp(-1, 1)

        res = torch.abs(imgs - imgs_w.detach())
        res_output = res.clone()
        for i in range(res_output.shape[0]):
            min_val, max_val = torch.min(res_output[i]), torch.max(res_output[i])
            res_output[i] = (res_output[i] - min_val) / (max_val - min_val)

        psnr = - kornia.losses.psnr_loss(imgs_w.detach(), imgs.detach(), max_val=2.0).item()
        ssim = 1 - 2 * kornia.losses.ssim_loss(imgs_w.detach(), imgs, window_size=11, reduction="mean")
        metric_logger.update(psnr=float(psnr), ssim=float(ssim))

        discriminator.requires_grad_(True)
        d_label_cover = discriminator(imgs)
        d_label_encoded = discriminator(imgs_w.detach())
        d_loss = criterion_MSE(d_label_cover - torch.mean(d_label_encoded),
                               label_cover * torch.ones_like(d_label_cover)) + \
                 criterion_MSE(d_label_encoded - torch.mean(d_label_cover),
                               label_encoded * torch.ones_like(d_label_encoded))
        d_loss = d_loss / accumulation_steps
        d_loss.backward()
        discriminator.requires_grad_(False)

        imgs_benign, imgs_mal = watermark_net.augmenter(imgs_w, imgs, attrs)

        hat_bg_message_b, hat_face_message_b = watermark_net.extractor(imgs_benign, masks)
        hat_bg_message_m, hat_face_message_m = watermark_net.extractor(imgs_mal, masks)

        mse_loss = criterion_MSE(imgs_w, imgs)
        lpips_loss = criterion_LPIPS(imgs_w, imgs).mean()
        freq_loss = criterion_freq(imgs_w, imgs, prior)
        L_img = 1.0 * mse_loss + 0.5 * lpips_loss + 2 * freq_loss

        # 水印监督
        benign_loss = criterion_MSE(hat_bg_message_b, message_bg) + criterion_MSE(hat_face_message_b, message_face)
        malicious_loss = criterion_MSE(hat_bg_message_m, message_bg) + criterion_MSE(hat_face_message_m,
                                                                                     torch.zeros_like(message_face))
        L_dec = benign_loss + malicious_loss

        g_label_cover = discriminator(imgs)
        g_label_encoded = discriminator(imgs_w)
        g_loss_on_discriminator = criterion_MSE(g_label_cover - torch.mean(g_label_encoded),
                                                label_encoded * torch.ones_like(g_label_cover)) + \
                                  criterion_MSE(g_label_encoded - torch.mean(g_label_cover),
                                                label_cover * torch.ones_like(g_label_encoded))

        L_total = 20.0 * L_dec + 1.0 * L_img + 0.1 * g_loss_on_discriminator
        #
        L_total = L_total / accumulation_steps
        L_total.backward()
        if (it + 1) % accumulation_steps == 0:
            optimizer_emb.step()
            optimizer_ext.step()
            optimizer_dis.step()
            optimizer_emb.zero_grad(set_to_none=True)
            optimizer_ext.zero_grad(set_to_none=True)
            optimizer_dis.zero_grad(set_to_none=True)

        ber_bg_b = decoded_message_error_rate_batch(message_bg, hat_bg_message_b)
        metric_logger.update(ber_bg_b=ber_bg_b)
        ber_bg_m = decoded_message_error_rate_batch(message_bg, hat_bg_message_m)
        metric_logger.update(ber_bg_m=ber_bg_m)
        ber_face_b = decoded_message_error_rate_batch(message_face, hat_face_message_b)
        metric_logger.update(ber_face_b=ber_face_b)
        ber_face_m = decoded_message_error_rate_batch(message_face, hat_face_message_m)
        metric_logger.update(ber_face_m=ber_face_m)

        metric_logger.update(mse_loss=float(mse_loss.detach()),
                             lpips_loss=float(lpips_loss.detach()),
                             freq_loss=float(freq_loss.detach()),
                             d_loss=float(d_loss.detach()),
                             g_loss=float(g_loss_on_discriminator.detach()),
                             L_dec=float(L_dec.detach()),
                             total_loss=float(L_total.detach()))

        if it % 500 == 0:
            save_image((imgs + 1) / 2, os.path.join(args.output_dir, f'{epoch:03}_{it:03}_train_0_ori.png'), nrow=8)
            save_image((imgs_w + 1) / 2, os.path.join(args.output_dir, f'{epoch:03}_{it:03}_train_1_w.png'), nrow=8)
            save_image((imgs_mal + 1) / 2, os.path.join(args.output_dir, f'{epoch:03}_{it:03}_train_2_mal.png'), nrow=8)
            save_image(res_output, os.path.join(args.output_dir, f'{epoch:03}_{it:03}_train_3_res.png'), nrow=8)

    if (it + 1) % accumulation_steps != 0:
        optimizer_emb.step()
        optimizer_ext.step()
        optimizer_dis.step()
        optimizer_emb.zero_grad(set_to_none=True)
        optimizer_ext.zero_grad(set_to_none=True)
        optimizer_dis.zero_grad(set_to_none=True)
    return 0


@torch.no_grad()
def validate_model(watermark_net: WatermarkNet,
                   val_loader,
                   epoch: int,
                   device: torch.device):
    watermark_net.eval()

    header = f'val'

    with torch.no_grad():
        for (imgs, masks, attrs) in tqdm(val_loader, total=len(val_loader), desc=header, ncols=100):
            imgs = imgs.to(device)
            B = imgs.size(0)
            masks = masks.to(device)
            message_bg = (
                    (torch.randint(0, 2, (imgs.shape[0], args.message_length), device=device).float() * 2 - 1)
                    * args.message_range
            )
            message_face = identity_maker(imgs)

            out = watermark_net(imgs, message_bg, message_face, masks, attrs)
            imgs_w = out["imgs_w"].detach()


            psnr_val = float(-kornia.losses.psnr_loss(imgs_w, imgs, 2).item())
            ssim_val = float(1.0 - 2.0 * kornia.losses.ssim_loss(
                (imgs_w + 1) / 2, (imgs + 1) / 2, window_size=11, reduction="mean"
            ).item())

            ber_bg_b = decoded_message_error_rate_batch(message_bg, out["hat_bg_message_b"])
            ber_bg_m = decoded_message_error_rate_batch(message_bg, out["hat_bg_message_m"])
            ber_face_b = decoded_message_error_rate_batch(message_face, out["hat_face_message_b"])
            ber_face_m = decoded_message_error_rate_batch(message_face, out["hat_face_message_m"])

            print(f"PSNR: {float(psnr_val):.2f} | "
                  f"ssim: {float(ssim_val):.4f} | "
                  f"ber_bg_b: {float(ber_bg_b):.4f} | "
                  f"ber_bg_m: {float(ber_bg_m):.4f} | "
                  f"ber_face_b: {float(ber_face_b):.4f} | "
                  f"ber_face_m: {float(ber_face_m):.4f} | "
                  )
    return 0


def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]

    message = message.gt(0.0)
    decoded_message = decoded_message.gt(0.0)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate


def save_checkpoint(watermark_net,
                    optimizer_dis,
                    discriminator,
                    optimizer_emb,
                    optimizer_ext,
                    epoch, path):
    torch.save({
        'epoch': epoch,
        'watermark_net_state_dict': watermark_net.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),

        # 保存优化器状态
        'optimizer_emb_state_dict': optimizer_emb.state_dict(),
        'optimizer_ext_state_dict': optimizer_ext.state_dict(),
        'optimizer_dis_state_dict': optimizer_dis.state_dict(),
    }, path)


class PerceptualMaskExtractor(nn.Module):

    def __init__(
            self,
            kernel_sizes=(3, 5, 9),
            kernel_weights=(0.5, 0.3, 0.2),
            fuse_weights=(0.45, 0.25, 0.30),  # std, relative contrast, gradient
            q_low=0.02,
            q_high=0.98,
            smooth_kernel=5,
            eps=1e-6
    ):
        super().__init__()

        assert len(kernel_sizes) == len(kernel_weights)
        assert len(fuse_weights) == 3

        wsum = sum(kernel_weights)
        self.kernel_sizes = kernel_sizes
        self.kernel_weights = [w / wsum for w in kernel_weights]

        fsum = sum(fuse_weights)
        self.fuse_weights = [w / fsum for w in fuse_weights]

        self.q_low = q_low
        self.q_high = q_high
        self.smooth_kernel = smooth_kernel
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _to_01(self, images):
        if images.min() < -0.1:
            images = (images + 1.0) / 2.0
        return images.clamp(0.0, 1.0)

    def _reflect_avg_pool(self, x, k):
        p = k // 2
        x = F.pad(x, (p, p, p, p), mode="reflect")
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=0)

    def _local_mean_std(self, x, k):
        mean = self._reflect_avg_pool(x, k)
        mean_sq = self._reflect_avg_pool(x * x, k)
        var = (mean_sq - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var + self.eps)
        return mean, std

    def _robust_norm(self, x):
        B = x.shape[0]
        flat = x.flatten(1)

        lo = torch.quantile(flat, self.q_low, dim=1, keepdim=True).view(B, 1, 1, 1)
        hi = torch.quantile(flat, self.q_high, dim=1, keepdim=True).view(B, 1, 1, 1)

        x = (x - lo) / (hi - lo + self.eps)
        return x.clamp(0.0, 1.0)

    def _grad_mag(self, gray):
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + self.eps)

    def _smooth(self, x, k):
        if k <= 1:
            return x
        return self._reflect_avg_pool(x, k)

    def forward(self, images):
        img = self._to_01(images)

        gray = (
                0.299 * img[:, 0:1] +
                0.587 * img[:, 1:2] +
                0.114 * img[:, 2:3]
        )

        ms_std = 0.0
        ms_rel_contrast = 0.0

        for k, w in zip(self.kernel_sizes, self.kernel_weights):
            mean, std = self._local_mean_std(gray, k)

            ms_std = ms_std + w * std

            rel_contrast = std / (mean.abs() + 0.05)
            ms_rel_contrast = ms_rel_contrast + w * rel_contrast

        grad = self._grad_mag(gray)

        ms_std = self._robust_norm(ms_std)
        ms_rel_contrast = self._robust_norm(ms_rel_contrast)
        grad = self._robust_norm(grad)

        a, b, c = self.fuse_weights
        prior = a * ms_std + b * ms_rel_contrast + c * grad

        prior = self._smooth(prior, self.smooth_kernel)
        prior = self._robust_norm(prior)

        return prior


class PerceptualDTCWTLoss(nn.Module):

    def __init__(
            self,
            level_weights=(1.0, 0.5, 0.25),
            lambda_high=0.02,

            # low-frequency weighting
            low_w_min=0.75,
            low_w_max=2.50,
            low_gamma=1.20,

            # high-frequency weighting (very mild)
            high_w_min=1.00,
            high_w_max=1.15,
            high_gamma=1.00,

            # log penalty scale for high-frequency magnitude
            high_log_scale=20.0,

            eps=1e-6,
            device="cuda"
    ):
        super().__init__()

        self.level_weights = level_weights
        self.lambda_high = lambda_high

        self.low_w_min = low_w_min
        self.low_w_max = low_w_max
        self.low_gamma = low_gamma

        self.high_w_min = high_w_min
        self.high_w_max = high_w_max
        self.high_gamma = high_gamma

        self.high_log_scale = high_log_scale
        self.eps = eps

        self.dtcwt = DTCWTForward(
            J=1,
            biort="near_sym_a",
            qshift="qshift_a"
        ).to(device)
        self.dtcwt.requires_grad_(False)

    def _make_weight(
            self, prior, size, w_min, w_max, gamma, clamp_min=0.5, clamp_max=4.0
    ):
        w = F.interpolate(prior, size=size, mode="bilinear", align_corners=False)

        w = w_min + (w_max - w_min) * (1.0 - w).pow(gamma)

        w = w / (w.mean(dim=(2, 3), keepdim=True) + self.eps)

        w = w.clamp(clamp_min, clamp_max)
        return w

    def _weighted_mean(self, x, w):
        """
        x: [B,C,H,W] or [B,C,D,H,W]
        w: [B,1,H,W] or [B,1,1,H,W]
        """
        return (x * w).mean()

    def forward(self, img_w, img_orig, visibility_prior=None):
        """
        img_w: [B,3,H,W]
        img_orig: [B,3,H,W]
        visibility_prior: [B,1,H,W] in [0,1], 值越大越可遮蔽
        """
        residual = img_w - img_orig

        total_low = 0.0
        total_high = 0.0

        x = residual

        for lw in self.level_weights:
            Yl, Yh = self.dtcwt(x)

            low_mag = torch.sqrt(Yl * Yl + self.eps)

            if visibility_prior is not None:
                w_low = self._make_weight(
                    visibility_prior,
                    size=Yl.shape[-2:],
                    w_min=self.low_w_min,
                    w_max=self.low_w_max,
                    gamma=self.low_gamma
                )
                low_loss = self._weighted_mean(low_mag, w_low)
            else:
                low_loss = low_mag.mean()

            total_low = total_low + lw * low_loss

            scale_h = Yh[0]
            real = scale_h[..., 0]
            imag = scale_h[..., 1]

            high_mag = torch.sqrt(real * real + imag * imag + self.eps)

            high_term = torch.log1p(self.high_log_scale * high_mag) / math.log1p(self.high_log_scale)

            if visibility_prior is not None and self.high_w_max > self.high_w_min:
                w_high = self._make_weight(
                    visibility_prior,
                    size=(scale_h.shape[3], scale_h.shape[4]),
                    w_min=self.high_w_min,
                    w_max=self.high_w_max,
                    gamma=self.high_gamma
                ).unsqueeze(2)  # [B,1,1,H,W]

                high_loss = self._weighted_mean(high_term, w_high)
            else:
                high_loss = high_term.mean()

            total_high = total_high + lw * high_loss

            x = Yl

        total = total_low + self.lambda_high * total_high

        return total


if __name__ == '__main__':
    import lpips

    criterion_lpips = lpips.LPIPS().to("cuda")
    prior_extractor = PerceptualMaskExtractor().to(device)
    args = parser.parse_args()
    identity_maker = IdentityWatermark(
        n_bits=args.message_length,
        message_range=args.message_range,
        image_range="-1_1",
        input_size=112,
        vote_k=5,
        ctx_id=0,
        seed=2026,
    )
    main(args)
