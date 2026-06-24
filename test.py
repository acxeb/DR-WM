import argparse
import os
import sys
from tqdm import tqdm
import omegaconf
from modules.logger import MetricLogger, redirect_stdout_to_file
import matplotlib
matplotlib.use('TkAgg')
import kornia
from net_project1 import WatermarkNet
from noise.augmenter_fix import Augmenter
from torch.utils.data import DataLoader
from modules.dataloader import ImageMaskAttrDataset
from modules.identity_maker import IdentityWatermark
from modules.Encoder_U import DW_Encoder
from modules.Decoder_U import DW_Decoder
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

parser = argparse.ArgumentParser()

group = parser.add_argument_group('Experiments parameters')
group.add_argument("--batch_size", type=int, default=8, required=False)
group.add_argument("--test_dir", type=str, default="/root/autodl-tmp/CelebAMask-HQ/celeba-hq/256/test_256", required=False)
group.add_argument("--masks_dir", type=str, default="/root/autodl-tmp/CelebAMask-HQ/celeba-hq/256/mask_test_256", required=False)
group.add_argument("--message_length", type=int, default=64, required=False)
group.add_argument("--message_range", type=float, default=0.1, required=False)
group.add_argument("--output_dir", type=str, default="output/test/", help="Output directory for logs and images (Default: /output)")
group.add_argument("--checkpoint_dir", type=str, default="checkpoint_dir/checkpoint_epoch_10.pth")
group.add_argument("--augmenter_config", type=str, default="C:/Users/22898/PycharmProjects/DualWatermarkNet/config/augs_test.yaml", help="Path to the augmenter config file")




def main(arg):
    import lpips
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    augmenter_cfg = omegaconf.OmegaConf.load(args.augmenter_config)
    augmenter = Augmenter(augmenter_cfg["augs"], augmenter_cfg["augs_params"]).to(device)
    print("Augmenter:", augmenter)
    test_dataset = ImageMaskAttrDataset(args.test_dir, args.masks_dir, image_transform=transform,
                                         mask_transform=mask_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    embedder = DW_Encoder(args.message_length, args.message_length).to(device)
    extractor = DW_Decoder(args.message_length, args.message_length).to(device)

    net = WatermarkNet(embedder=embedder, extractor=extractor, augmenter=augmenter)
    net = net.to(device)

    checkpoint = torch.load(arg.checkpoint_dir, weights_only=True)
    state_dict = checkpoint['watermark_net_state_dict']
    net.load_state_dict(state_dict)
    criterion_LPIPS = lpips.LPIPS().to("cuda")

    log_file = redirect_stdout_to_file(list(augmenter_cfg["augs"].keys())[0], log_dir="logs/test/", is_train=False)
    original_stdout = sys.stdout
    sys.stdout = log_file

    val_metrics = {
        "ber_bg_b": 0.0,
        "ber_bg_m": 0.0,
        "ber_face_b": 0.0,
        "ber_face_m": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
        "score_b": 0.0,
        "score_m": 0.0,
    }
    total_samples = 0

    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc="test")
    for step, (imgs, masks, attrs) in enumerate(pbar, 1):
        net.eval()
        imgs = imgs.to(device)
        masks = masks.to(device)

        B = imgs.shape[0]
        total_samples += B

        with torch.no_grad():
            message_bg = (
                    (torch.randint(0, 2, (imgs.shape[0], args.message_length), device=device).float() * 2 - 1)
                    * args.message_range
            )
            # message_face = (
            #         (torch.randint(0, 2, (imgs.shape[0], args.message_length), device=device).float() * 2 - 1)
            #         * args.message_range
            # )
            message_face = identity_maker(imgs)

            delta_bg, delta_face = net.embedder(imgs, message_bg, message_face, masks)
            imgs_w = (imgs + delta_bg + delta_face).clamp(-1, 1)
            imgs_benign, imgs_mal = net.augmenter(imgs_w, imgs, attrs)
            message_face_noised_b = identity_maker(imgs_benign)
            message_face_noised_m = identity_maker(imgs_mal)

            hat_bg_message_b, hat_face_message_b = net.extractor(imgs_benign, masks)
            hat_bg_message_m, hat_face_message_m = net.extractor(imgs_mal, masks)

        ber_bg_b = decoded_message_error_rate_batch(message_bg, hat_bg_message_b)
        ber_bg_m = decoded_message_error_rate_batch(message_bg, hat_bg_message_m)
        ber_face_b = decoded_message_error_rate_batch(message_face, hat_face_message_b)
        ber_face_m = decoded_message_error_rate_batch(message_face, hat_face_message_m)
        psnr = - kornia.losses.psnr_loss(imgs_w.detach(), imgs.detach(), max_val=2.0).item()
        ssim_val = 1 - 2 * kornia.losses.ssim_loss(imgs_w.detach(), imgs, window_size=11, reduction="mean").item()
        lpips_val = criterion_LPIPS(imgs_w.detach(), imgs.detach()).mean().item()
        score_b = identity_watermark_score(hat_face_message_b, message_face_noised_b)
        score_m = identity_watermark_score(hat_face_message_m, message_face_noised_m)

        val_metrics["ber_bg_b"] += float(ber_bg_b) * B
        val_metrics["ber_bg_m"] += float(ber_bg_m) * B
        val_metrics["ber_face_b"] += float(ber_face_b) * B
        val_metrics["ber_face_m"] += float(ber_face_m) * B
        val_metrics["psnr"] += float(psnr) * B
        val_metrics["ssim"] += float(ssim_val) * B
        val_metrics["lpips"] += float(lpips_val) * B
        val_metrics["score_b"] += score_b.sum().item()
        val_metrics["score_m"] += score_m.sum().item()

        res = (imgs_w.detach() - imgs.detach()).abs()
        for i in range(res.shape[0]):
            min_val = res[i].min()
            max_val = res[i].max()
            if max_val > min_val:
                res[i] = (res[i] - min_val) / (max_val - min_val)
            else:
                res[i] = 0.0

        res_benign = (imgs_benign.detach() - imgs.detach()).abs()
        for i in range(res_benign.shape[0]):
            min_val = res_benign[i].min()
            max_val = res_benign[i].max()
            if max_val > min_val:
                res_benign[i] = (res_benign[i] - min_val) / (max_val - min_val)
            else:
                res_benign[i] = 0.0

        res_mal = (imgs_mal.detach() - imgs.detach()).abs()
        for i in range(res_mal.shape[0]):
            min_val = res_mal[i].min()
            max_val = res_mal[i].max()
            if max_val > min_val:
                res_mal[i] = (res_mal[i] - min_val) / (max_val - min_val)
            else:
                res_mal[i] = 0.0

        # if step == 500:
        #     save_image((imgs+1)/2,
        #                os.path.join(args.output_dir, f'{step:03}_test_{list(augmenter_cfg["augs"].keys())[0]}_0_ori.png'),
        #                nrow=8)
        #     save_image((imgs_w+1)/2,
        #                os.path.join(args.output_dir, f'{step:03}_test_{list(augmenter_cfg["augs"].keys())[0]}_1_w.png'),
        #                nrow=8)
        #     save_image((imgs_benign+1)/2,
        #                os.path.join(args.output_dir, f'{step:03}_test_{list(augmenter_cfg["augs"].keys())[0]}_2_benign.png'),
        #                nrow=8)
        #     save_image((imgs_mal + 1) / 2,
        #                os.path.join(args.output_dir, f'{step:03}_test_{list(augmenter_cfg["augs"].keys())[0]}_3_mal.png'),
        #                nrow=8)
        #     save_image(res,
        #                os.path.join(args.output_dir,
        #                             f'{step:03}_test_{list(augmenter_cfg["augs"].keys())[0]}_6_res.png'),
        #                nrow=8)
        #     save_image(res_benign,
        #                os.path.join(args.output_dir,
        #                             f'{step:03}_test_{list(augmenter_cfg["augs"].keys())[0]}_7_res_benign.png'),
        #                nrow=8)
        #     save_image(res_mal,
        #                os.path.join(args.output_dir,
        #                             f'{step:03}_test_{list(augmenter_cfg["augs"].keys())[0]}_8_res_mal.png'),
        #                nrow=8)

    print("-" * 50)
    print(f"Final Validation Results over {total_samples} samples:")
    for key in val_metrics:
        val_metrics[key] /= total_samples
        print(f"{key}: {val_metrics[key]:.6f}")
    print("-" * 50)

    sys.stdout = original_stdout
    if hasattr(log_file, 'close'):
        log_file.close()

def identity_watermark_score(
    hat_message_face: torch.Tensor,
    current_message_face: torch.Tensor,
):
    pred_bits = hat_message_face > 0
    ref_bits = current_message_face > 0

    score = (pred_bits == ref_bits).float().mean(dim=1)

    return score

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


if __name__ == '__main__':
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
