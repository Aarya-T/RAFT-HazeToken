# test_dataloader.py

from argparse import Namespace
from core.datasets import fetch_dataloader

args = Namespace(
    stage='kitti_hazy',
    haze_level='light',
    image_size=[384,512],
    batch_size=2
)

loader = fetch_dataloader(args)

batch = next(iter(loader))

for i, item in enumerate(batch):
    print(i, item.shape)