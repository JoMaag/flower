"""
Utility functions for FedPG-BR.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from torch.nn import DataParallel


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional file path for logging
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    
    # Suppress Flower's duplicate logging and deprecation warnings
    logging.getLogger("flwr").setLevel(logging.WARNING)
    
    # Suppress Ray's verbose logging
    logging.getLogger("ray").setLevel(logging.WARNING)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_inner_model(model):
    """Unwrap DataParallel if needed."""
    return model.module if isinstance(model, DataParallel) else model


def move_to(var, device: str):
    """Move variable(s) to device."""
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return [move_to(v, device) for v in var]
    elif hasattr(var, 'to'):
        return var.to(device)
    return var


def save_checkpoint(
    path: str,
    policy,
    optimizer,
    round_num: int,
    metrics: Optional[Dict[str, Any]] = None,
):
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "round": round_num,
        "policy_state_dict": get_inner_model(policy).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "metrics": metrics or {},
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    policy,
    optimizer=None,
    device: str = "cpu",
) -> int:
    """
    Load training checkpoint.
    
    Returns:
        Round number from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    
    get_inner_model(policy).load_state_dict(checkpoint["policy_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])
    
    return checkpoint.get("round", 0)
