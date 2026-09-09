"""Optimizer group helpers for scale-oriented OpenWAM training."""

from __future__ import annotations


def build_trainable_parameters(model, *, action_lr=None, video_lr=None, la_lr=None):
    """Bucket every trainable top-level param by owner, then build optimizer groups:

    - ``video_backbone`` → optional ``video_lr``
    - ``action_backbone`` → optional ``action_lr``
    - ``latent_action_expert`` → optional ``la_lr`` (la_tri_system's supervised
      latent-action expert)
    - every other trainable top-level module → the optimizer base
      ``learning_rate``, never a per-group override.

    Source of truth: ``BaseWAMArchitecture.get_trainable_modules()`` (top-level
    children that hold at least one ``requires_grad`` param). Only ``requires_grad``
    params are collected, so the model yaml ``freeze:`` list — which flips
    ``requires_grad`` — is the single gate for trainability; frozen params never
    reach the optimizer. With no per-group LR override, returns a flat param list;
    otherwise returns groups.
    """
    action_params, video_params, la_params, other_params = [], [], [], []
    for mod_name, mod in model.architecture.get_trainable_modules().items():
        if mod_name == "video_backbone":
            bucket = video_params
        elif mod_name == "action_backbone":
            bucket = action_params
        elif mod_name == "latent_action_expert":
            bucket = la_params
        else:
            bucket = other_params
        bucket.extend(param for param in mod.parameters() if param.requires_grad)

    if action_lr is None and video_lr is None and la_lr is None:
        return action_params + video_params + la_params + other_params

    # action / video / la may carry their own LR; "other" always rides the base LR (lr=None).
    groups = []
    for params, lr in [
        (action_params, action_lr),
        (video_params, video_lr),
        (la_params, la_lr),
        (other_params, None),
    ]:
        if not params:
            continue
        group = {"params": params}
        if lr is not None:
            group["lr"] = lr
        groups.append(group)
    return groups


__all__ = ["build_trainable_parameters"]
