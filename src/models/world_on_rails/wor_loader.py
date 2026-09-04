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
    freeze_backbone: bool = True,
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

        # Training with a frozen backbone writes split checkpoints: the unchanging
        # vision weights land once in a sibling frozen_backbone.pth and the per-epoch
        # files carry only the trained heads. Reunite them here so callers still get a
        # complete model from a single path.
        if isinstance(checkpoint, dict) and checkpoint.get("partial"):
            frozen_path = os.path.join(
                os.path.dirname(os.path.abspath(checkpoint_path)),
                checkpoint.get("frozen_ref", "frozen_backbone.pth")
            )
            if os.path.exists(frozen_path):
                frozen = torch.load(frozen_path, map_location="cpu")
                frozen_sd = frozen.get("model", frozen)
                state_dict = {**frozen_sd, **state_dict}
                print(f"✓ Merged {len(frozen_sd)} frozen backbone tensors from: {frozen_path}")
            else:
                # Not fatal: the backbone falls back to whatever the model was built
                # with (ImageNet, or --weights_path). Loud, because that silently
                # changes which perception stack the policy heads sit on.
                print(f"[Warning] Checkpoint is partial but {frozen_path} is missing - "
                      f"the vision backbone will keep its freshly initialized weights, "
                      f"which likely does NOT match what these heads were trained on.")

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
