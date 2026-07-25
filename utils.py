"""
Utility Functions Module
Common helper functions for reproducibility, logging, and GPU memory monitoring.
"""
import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds across all libraries to ensure experimental reproducibility.
    :param seed: Global random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(log_path: str = None, name: str = "qlora_ad") -> logging.Logger:
    """
    Initialize a logger that outputs to both console and file.
    :param log_path: Path to save log file
    :param name: Logger name
    :return: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_gpu_memory_usage() -> str:
    """
    Get current GPU memory usage for debugging and logging.
    :return: Formatted memory usage string
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    return f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB"
"""
Utility Functions Module
Common helper functions for reproducibility, logging, and GPU memory monitoring.
"""
import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds across all libraries to ensure experimental reproducibility.
    :param seed: Global random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(log_path: str = None, name: str = "qlora_ad") -> logging.Logger:
    """
    Initialize a logger that outputs to both console and file.
    :param log_path: Path to save log file
    :param name: Logger name
    :return: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_gpu_memory_usage() -> str:
    """
    Get current GPU memory usage for debugging and logging.
    :return: Formatted memory usage string
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    return f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB"
