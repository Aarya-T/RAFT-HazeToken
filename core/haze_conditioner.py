import torch
import torch.nn as nn

from .vlm_tokens.clip_features import CLIPFeatureExtractor
from .vlm_tokens.global_token import GlobalHazeToken

class HazeConditioner(nn.Module):
    """
    Generates a Global Haze Token from Image 1.

    Pipeline:
        Image
          ↓
        Frozen CLIP
          ↓
      512-D Image Embedding
          ↓
     Global Haze Token (128-D)
    """

    def __init__(self):
        super().__init__()

        self.clip = CLIPFeatureExtractor()
        self.global_token = GlobalHazeToken()

    def forward(self, image):

        image_embedding = self.clip(image)

        haze_token = self.global_token(image_embedding)

        return haze_token