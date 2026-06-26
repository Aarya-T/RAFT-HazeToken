import torch
from argparse import Namespace

from core.raft import RAFT

device = "cuda" if torch.cuda.is_available() else "cpu"

args = Namespace(
    small=False,
    dropout=0,
    alternate_corr=False,
    mixed_precision=False
)

model = RAFT(args).to(device)

image1 = torch.randn(1, 3, 384, 512).to(device)
image2 = torch.randn(1, 3, 384, 512).to(device)

with torch.no_grad():
    output = model(image1, image2)

print("Forward pass successful!")
print("Number of predictions:", len(output))
print("Final output shape:", output[-1].shape)