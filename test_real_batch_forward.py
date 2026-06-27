from argparse import Namespace
import torch

from core.datasets import fetch_dataloader
from core.raft import RAFT

if __name__ == "__main__":

    args = Namespace(
        stage='kitti_hazy',
        haze_level='light',
        image_size=[384, 512],
        batch_size=2,

        # RAFT args
        small=False,
        mixed_precision=False,
        alternate_corr=False,

        # HazeToken args
        use_haze_token=True
    )

    print("Creating dataloader...")
    loader = fetch_dataloader(args)

    print("Loading batch...")
    image1, image2, flow_gt, valid = next(iter(loader))

    print("Batch loaded:")
    print("image1:", image1.shape)
    print("image2:", image2.shape)

    print("Creating RAFT...")
    model = RAFT(args)

    if torch.cuda.is_available():
        model = model.cuda()
        image1 = image1.cuda()
        image2 = image2.cuda()

    model.eval()

    print("Running forward pass...")

    with torch.no_grad():
        flow_predictions = model(image1, image2)

    print("Forward pass successful!")
    print("Number of predictions:", len(flow_predictions))
    print("Final output shape:", flow_predictions[-1].shape)