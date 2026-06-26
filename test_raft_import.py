import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

import torch
from argparse import Namespace

from core.raft import RAFT

args = Namespace(
    small=False,
    dropout=0,
    alternate_corr=False,
    mixed_precision=False
)

print("Creating RAFT...")

model = RAFT(args)

print("RAFT created successfully!")

image1 = torch.randn(1, 3, 384, 512)
image2 = torch.randn(1, 3, 384, 512)

print("Running forward pass...")

with torch.no_grad():
    output = model(image1, image2)

print("Forward pass completed!")
print("Number of predictions:", len(output))
print("Final prediction shape:", output[-1].shape)