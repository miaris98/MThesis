"""World on Rails (WoR) Weight & Checkpoint Loader.

Supports loading official PCLA weights (wor_nc, wor_lb), ImageNet pretrained backbones,
and custom trained checkpoints with automatic state dict prefix stripping.
"""
from typing import Optional, Union
import os
import urllib.request
import torch
import torch.nn as nn

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy


# Public URLs or Zenodo links for pre-trained weights
PCLA_WEIGHT_URLS = {
    "wor_nc": "https://github.com/MasoudJTehrani/PCLA/releases/download/v1.0/wor_nc.pth",
    "wor_lb": "https://github.com/MasoudJTehrani/PCLA/releases/download/v1.0/wor_lb.pth"
}


def download_pretrained_weights(model_type: str = "wor_nc", save_dir: str = "weights") -> str:
    """
    Downloads pretrained WoR weights if not present locally.
    """
    os.makedirs(save_dir, exist_ok=True)
    target_path = os.path.join(save_dir, f"{model_type}.pth")
    if os.path.exists(target_path):
        print(f"✓ Found existing pretrained weights at: {target_path}")
        return target_path

    url = PCLA_WEIGHT_URLS.get(model_type)
    if url:
        print(f"--> Downloading pretrained {model_type} weights from {url}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"✓ Downloaded weights to: {target_path}")
            return target_path
        except Exception as e:
            print(f"[Warning] Could not download from {url}: {e}. Initializing model with ImageNet backbone.")
    return ""


def load_wor_model(
    checkpoint_path: Optional[str] = None,
    backbone_name: str = "resnet34",
    pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
    device: Union[str, torch.device] = "cpu"
) -> WorldOnRailsPolicy:
    """
    Instantiates and loads a World on Rails policy model.
    """
    model = WorldOnRailsPolicy(
        backbone_name=backbone_name,
        pretrained=pretrained_backbone,
        freeze_backbone=freeze_backbone
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"--> Loading World on Rails model weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

        clean_dict = {}
        for k, v in state_dict.items():
            clean_k = k
            for prefix in ["module.", "model.", "_orig_mod."]:
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            clean_dict[clean_k] = v

        missing, unexpected = model.load_state_dict(clean_dict, strict=False)
        print(f"✓ Loaded weights into WorldOnRailsPolicy (Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)})")
    else:
        print(f"--> Initialized WorldOnRailsPolicy with {backbone_name.upper()} (ImageNet Pretrained: {pretrained_backbone})")

    model.to(device)
    return model
