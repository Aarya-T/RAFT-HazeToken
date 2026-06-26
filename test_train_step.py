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
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Dummy inputs
image1 = torch.randn(1, 3, 384, 512, device=device)
image2 = torch.randn(1, 3, 384, 512, device=device)

# Dummy ground-truth flow
gt_flow = torch.randn(1, 2, 384, 512, device=device)

optimizer.zero_grad()

flow_predictions = model(image1, image2)

loss = torch.nn.functional.l1_loss(
    flow_predictions[-1],
    gt_flow
)

print("Loss:", loss.item())

loss.backward()

optimizer.step()

print("One training step completed successfully!")