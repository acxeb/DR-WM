from modules.Encoder_U import DW_Encoder
from modules.Decoder_U import DW_Decoder
from noise.augmenter import Augmenter

class WatermarkNet(nn.Module):
    def __init__(
            self,
            embedder: DW_Encoder,
            extractor: DW_Decoder,
            augmenter: Augmenter=None,
    ):
        super().__init__()
        self.embedder = embedder
        self.extractor = extractor
        self.augmenter = augmenter

    def forward(self, imgs: torch.Tensor, message_bg, message_face, masks, attrs=None):
        bsz = imgs.shape[0]
        delta_bg, delta_face = self.embedder(imgs, message_bg, message_face, masks)
        imgs_w = (imgs + delta_bg + delta_face).clamp(-1, 1)
        imgs_benign, imgs_mal = self.augmenter(imgs_w, imgs, attrs)
        hat_bg_message_b, hat_face_message_b = self.extractor(imgs_benign, masks)
        hat_bg_message_m, hat_face_message_m = self.extractor(imgs_mal, masks)
        return {
            "imgs_w": imgs_w,
            "imgs_b": imgs_benign,
            "imgs_m": imgs_mal,
            "hat_bg_message_b": hat_bg_message_b,
            "hat_face_message_b": hat_face_message_b,
            "hat_bg_message_m": hat_bg_message_m,
            "hat_face_message_m": hat_face_message_m,
        }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")