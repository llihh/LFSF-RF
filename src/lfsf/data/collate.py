"""Collate functions for MFIF data loading."""

import torch


def collate_mfif_teacher(batch):
    max_stack = max(item["stack"].shape[0] for item in batch)
    bsz = len(batch)
    _, h, w = batch[0]["f0"].shape
    stack = torch.zeros((bsz, max_stack, 1, h, w), dtype=torch.float32)
    stack_mask = torch.zeros((bsz, max_stack), dtype=torch.float32)
    names = []
    for i, item in enumerate(batch):
        n = item["stack"].shape[0]
        stack[i, :n] = item["stack"]
        stack_mask[i, :n] = 1.0
        names.append(item["name"])
    return {
        "name": names,
        "stack": stack,
        "stack_mask": stack_mask,
        "f0": torch.stack([item["f0"] for item in batch], dim=0),
        "uncertainty": torch.stack([item["uncertainty"] for item in batch], dim=0),
        "confidence": torch.stack([item["confidence"] for item in batch], dim=0),
        "teacher": torch.stack([item["teacher"] for item in batch], dim=0),
    }


def collate_mfif_inference(batch):
    max_stack = max(item["stack"].shape[0] for item in batch)
    bsz = len(batch)
    _, h, w = batch[0]["f0"].shape
    stack = torch.zeros((bsz, max_stack, 1, h, w), dtype=torch.float32)
    stack_mask = torch.zeros((bsz, max_stack), dtype=torch.float32)
    names = []
    for i, item in enumerate(batch):
        n = item["stack"].shape[0]
        stack[i, :n] = item["stack"]
        stack_mask[i, :n] = 1.0
        names.append(item["name"])
    return {
        "name": names,
        "stack": stack,
        "stack_mask": stack_mask,
        "f0": torch.stack([item["f0"] for item in batch], dim=0),
        "uncertainty": torch.stack([item["uncertainty"] for item in batch], dim=0),
        "confidence": torch.stack([item["confidence"] for item in batch], dim=0),
    }
