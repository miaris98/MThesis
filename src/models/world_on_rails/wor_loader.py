"""World on Rails (WoR) Weight & Checkpoint Loader.

Supports loading official PCLA weights (wor_nc, wor_lb), ImageNet pretrained backbones,
and custom trained checkpoints with automatic state dict prefix stripping.
"""
from typing import Optional, Union
import os
import torch
import torch.nn as nn

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy


# The actual pretrained-WoR checkpoints live in the MasoudJTehrani/PCLA HuggingFace
# *dataset* repo (not a GitHub release - that v1.0/wor_nc.pth URL 404s, there was never
# a release with per-agent files). agents.json in the vendored Carla-utils/PCLA points
# "wor"/"nc" and "wor"/"lb" at these exact filenames.
PCLA_HF_REPO = "MasoudJTehrani/PCLA"
PCLA_HF_WEIGHT_FILES = {
    "wor_nc": "wor_pretrained/nocrash_weights/main_model_16.th",
    "wor_lb": "wor_pretrained/leaderboard_weights/main_model_10.th"
}


def download_pretrained_weights(model_type: str = "wor_nc", save_dir: str = "weights") -> str:
    """
    Downloads the pretrained WoR checkpoint if not present locally. This is the
    original WoR paper's own CARLA-domain-pretrained CameraModel (backbone_wide =
    resnet34), fetched from its HuggingFace dataset repo - use this instead of
    chasing other CARLA-pretrained sources, since its ResNet34 is layer-for-layer
    the same torchvision-style architecture PretrainedVisionEncoder uses (unlike
    e.g. TransFuser++/garage_2 checkpoints, which use a RegNet backbone and can
    only ever partially key-match this project's ResNet encoder).
    """
    os.makedirs(save_dir, exist_ok=True)
    target_path = os.path.join(save_dir, f"{model_type}.pth")
    if os.path.exists(target_path):
        print(f"✓ Found existing pretrained weights at: {target_path}")
        return target_path

    hf_filename = PCLA_HF_WEIGHT_FILES.get(model_type)
    if not hf_filename:
        print(f"[Warning] No known HuggingFace weight file for model_type='{model_type}'. "
              f"Initializing model with ImageNet backbone.")
        return ""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[Warning] huggingface_hub is not installed (pip install huggingface_hub) - "
              "cannot download pretrained WoR weights. Initializing model with ImageNet backbone.")
        return ""

    print(f"--> Downloading pretrained {model_type} weights from "
          f"hf://datasets/{PCLA_HF_REPO}/{hf_filename}...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=PCLA_HF_REPO, repo_type="dataset", filename=hf_filename
        )
        # hf_hub_download returns a path inside the HF cache - copy it into save_dir
        # so callers (and --weights_path) get the stable, predictable path they expect.
        import shutil
        shutil.copy(downloaded_path, target_path)
        print(f"✓ Downloaded weights to: {target_path}")
        return target_path
    except Exception as e:
        print(f"[Warning] Could not download {hf_filename} from {PCLA_HF_REPO}: {e}. "
              f"Initializing model with ImageNet backbone.")
        return ""


def load_wor_model(
    checkpoint_path: Optional[str] = None,
    backbone_name: str = "resnet34",
    pretrained_backbone: bool = True,
    freeze_backbone: bool = True,
    device: Union[str, torch.device] = "cpu",
    policy_arch: str = "cnn",
    route_points: int = 4
) -> Union[WorldOnRailsPolicy, QwenWorldOnRailsPolicy]:
    """
    Instantiates and loads a World on Rails policy model. `policy_arch` selects the
    decision head: "cnn" is the original SpatialQHead, "qwen100m"/"qwen500m"/"qwen900m"
    is the Qwen transformer trunk (see qwen_wor_policy.py) - both share the same
    checkpoint format (a WorldOnRailsTrainer run against either produces a state_dict
    keyed the same way, "encoder." for the frozen backbone), so this only needs to
    pick which class to instantiate before loading.
    """
    if policy_arch == "cnn":
        model = WorldOnRailsPolicy(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone
        )
    else:
        model = QwenWorldOnRailsPolicy(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            route_points=route_points,
            model_size=policy_arch.replace("qwen", "")
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
