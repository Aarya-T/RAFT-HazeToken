import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)

    Applies:
        output = gamma * feature + beta

    where gamma and beta are generated
    from the global haze token.
    """

    def __init__(self,
                 token_dim=128,
                 feature_dim=256):
        super().__init__()

        self.gamma = nn.Linear(token_dim, feature_dim)
        self.beta = nn.Linear(token_dim, feature_dim)

    def forward(self, feature, haze_token):

        # Generate channel-wise gamma and beta
        gamma = self.gamma(haze_token)
        beta = self.beta(haze_token)

        # Expand to match feature map dimensions
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM
        output = gamma * feature + beta

        return output