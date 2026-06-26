import torch

from core.vlm_tokens.film import FiLM

# Dummy context feature map (same shape as RAFT cnet output)
feature = torch.randn(1, 256, 54, 120)

# Dummy global haze token
token = torch.randn(1, 128)

# Create FiLM
film = FiLM()

# Apply FiLM
output = film(feature, token)

print("Input Feature Shape :", feature.shape)
print("Haze Token Shape    :", token.shape)
print("Output Shape        :", output.shape)

print("\nFirst channel statistics:")
print("Input Mean :", feature.mean().item())
print("Output Mean:", output.mean().item())