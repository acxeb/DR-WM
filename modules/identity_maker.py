import os
import cv2
import numpy as np
import torch
import insightface


class IdentityWatermark:

    def __init__(
        self,
        n_bits=64,
        message_range=0.1,
        image_range="-1_1",
        input_size=112,
        vote_k=5,
        seed=2026,
        ctx_id=0,
        model_path=None,
        providers=None,
    ):
        self.n_bits = n_bits
        self.message_range = float(message_range)
        self.image_range = image_range
        self.input_size = input_size
        self.vote_k = vote_k
        self.seed = seed

        if self.image_range not in ["-1_1", "0_1"]:
            raise ValueError("image_range must be '-1_1' or '0_1'.")

        if vote_k % 2 == 0:
            raise ValueError("vote_k should be odd, e.g. 3, 5, or 7.")

        if model_path is None:
            model_path = os.path.join(
                os.path.expanduser("~"),
                ".insightface",
                "models",
                "buffalo_l",
                "w600k_r50.onnx"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Recognition model not found: {model_path}")

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.rec_model = insightface.model_zoo.get_model(
            model_path,
            providers=providers
        )
        self.rec_model.prepare(ctx_id=ctx_id)

        self.emb_dim = 512

        rng = np.random.default_rng(seed)
        R = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(self.emb_dim, self.n_bits, self.vote_k)
        ).astype(np.float32)

        R = R / (np.linalg.norm(R, axis=0, keepdims=True) + 1e-12)

        self.R = R

    def _imgs_to_bgr_uint8(self, imgs: torch.Tensor):
        assert imgs.ndim == 4 and imgs.size(1) == 3, \
            f"Expected imgs [B,3,H,W], got {imgs.shape}"

        x = imgs.detach().float().cpu()

        if self.image_range == "-1_1":
            x = (x + 1.0) * 0.5

        x = x.clamp(0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8)

        bgr_imgs = []

        for i in range(x.size(0)):
            img_rgb = x[i].permute(1, 2, 0).numpy()
            img_bgr = img_rgb[..., ::-1].copy()

            if img_bgr.shape[0] != self.input_size or img_bgr.shape[1] != self.input_size:
                img_bgr = cv2.resize(
                    img_bgr,
                    (self.input_size, self.input_size),
                    interpolation=cv2.INTER_LINEAR
                )

            bgr_imgs.append(img_bgr)

        return bgr_imgs

    def _extract_embedding(self, img_bgr: np.ndarray):
        emb1 = self.rec_model.get_feat(img_bgr).astype(np.float32).reshape(-1)
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-12)

        img_flip = img_bgr[:, ::-1, :].copy()
        emb2 = self.rec_model.get_feat(img_flip).astype(np.float32).reshape(-1)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-12)

        emb = emb1 + emb2
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        return emb.astype(np.float32)

    def _embeddings_to_bits(self, embs: np.ndarray):

        logits = np.einsum("bd,dlk->blk", embs, self.R)

        signs = np.where(logits >= 0.0, 1.0, -1.0)

        vote = signs.sum(axis=2)  # [B,64]

        bits = np.where(vote >= 0.0, 1.0, -1.0).astype(np.float32)

        return bits

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor):
        device = imgs.device

        bgr_imgs = self._imgs_to_bgr_uint8(imgs)

        embs = []
        for img_bgr in bgr_imgs:
            emb = self._extract_embedding(img_bgr)
            embs.append(emb)

        embs = np.stack(embs, axis=0).astype(np.float32)

        bits = self._embeddings_to_bits(embs)

        message_face = torch.from_numpy(bits).to(
            device=device,
            dtype=torch.float32
        )

        message_face = message_face * self.message_range

        return message_face