import torch
import open_clip


class CLIPTextEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )

        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode_prompts(self, prompts):
        tokens = self.tokenizer(prompts).to(self.device)

        features = self.model.encode_text(tokens)

        features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu()