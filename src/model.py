"""Model factory built on `timm` (PyTorch Image Models)."""
import timm
import torch.nn as nn


def build_model(name: str = "tf_efficientnet_b4_ns",
                num_classes: int = 5,
                pretrained: bool = True,
                drop_rate: float = 0.3) -> nn.Module:
    """Create a classifier with an ImageNet-pretrained backbone.

    Any model string supported by timm works, e.g.:
        tf_efficientnet_b4_ns, convnext_small.fb_in22k, vit_base_patch16_384
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model
