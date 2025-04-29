from models.bottom_up_attention import BottomUpAttention
from datasets.vg import vg
import argparse
import torch
import numpy as np
import os
from datasets.vg_collator import VGCollator
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet101_Weights
from torchvision.models._utils import IntermediateLayerGetter
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--output-dir", type=str, default="bottom_up_features", required=False)
    parser.add_argument("--feature_type", type=str, default="resnet", required=False)
    parser.add_argument("--device", type=str, default="cpu", required=False)
    parser.add_argument("--num-workers", type=int, default=8, required=False)
    parser.add_argument('--vg-version',    default='1600-400-20',
                   help='Visual Genome version (vocabulary size)')
    parser.add_argument('--split',         default='train',
                   help='Dataset split to use: train/val/test/minival/minitrain')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    
    if args.feature_type == "bottom_up":
        model = BottomUpAttention()
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
    elif args.feature_type == "resnet":
        model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        model.eval()
        backbone = IntermediateLayerGetter(
            model,
            return_layers={'layer4':'0'}
        )
    else:
        raise ValueError(f"Invalid feature type: {args.feature_type}")

    # Load the dataset
    dataset = vg(args.vg_version, args.split)
    indices = list(range(len(dataset.image_index)))
    collator = VGCollator(dataset, device=args.device)
    dataloader = DataLoader(
        indices,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cuda')
    )
    # Generate features
    for i, (imgs, im_info, gt_targets) in enumerate(dataloader, 1):
        vg_dir, image_file_name = im_info[0][0], im_info[0][1]
        feature_dir = os.path.join(args.output_dir, args.feature_type, vg_dir)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir, exist_ok=True)
        if args.feature_type == "bottom_up":
            features = model(imgs, gt_targets)
            pool5_flat = features['pool5_flat']
        elif args.feature_type == "resnet":
            with torch.no_grad(): # Important for inference/feature extraction
                feats = backbone(imgs.tensors)
                pool5_flat = feats['0']
        print("features shape: ", pool5_flat.shape)
        feature_path = os.path.join(feature_dir, f"{image_file_name}.npy")
        print(f"Generated features to {feature_path}")
        np.save(feature_path, pool5_flat.cpu().numpy())

if __name__ == "__main__":
    main()
