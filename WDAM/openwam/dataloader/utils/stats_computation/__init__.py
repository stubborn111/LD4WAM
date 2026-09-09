"""Offline normalization-stats tools for the dataloaders.

CLI modules that scan a dataset once and write the stats the readers require
at train time:

  - ``robotwin_stats_computation`` — RoboTwin (``train.yaml``).
  - ``midtrain_stats_computation`` — multi-view robot buckets (``train_midtrain.yaml``).

Run them as ``python -m openwam.dataloader.utils.stats_computation.<module>``.
"""
