import torch

from core.haze_conditioner import HazeConditioner

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HazeConditioner().to(device)

image = torch.randn(1, 3, 384, 512).to(device)

token = model(image)

print("Token Shape:", token.shape)
print(token)