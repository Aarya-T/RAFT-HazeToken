import torch
import torch.nn as nn


class GlobalHazeToken(nn.Module):
    """
    Converts a global CLIP image embedding into
    a compact global haze token.
    """

    def __init__(self,
                 input_dim=512,
                 token_dim=128):
        super().__init__()

        self.projection = nn.Linear(input_dim, token_dim)

    def forward(self, image_embedding):

        token = self.projection(image_embedding)

        return token