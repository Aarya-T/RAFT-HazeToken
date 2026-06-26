import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class CLIPFeatureExtractor(nn.Module):
    """
    Frozen OpenCLIP image encoder.

    Input:
        image tensor (B,3,H,W)

    Output:
        normalized image embedding (B,512)
    """

    def __init__(self, device=None):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )

        self.model.eval()

        # Freeze CLIP
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

    @torch.no_grad()
    def forward(self, image):

        # Move to CLIP device
        image = image.to(self.device)

        # RAFT images are in [-1, 1]
        # Convert back to [0, 1]
        image = (image + 1.0) / 2.0

        # Resize for CLIP
        image = F.interpolate(
            image,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        # Normalize using CLIP statistics
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=image.device
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=image.device
        ).view(1, 3, 1, 1)

        image = (image - mean) / std

        features = self.model.encode_image(image)

        features = features / features.norm(dim=-1, keepdim=True)

        return features