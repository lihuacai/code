"""
Configuration Module
Centralized management of all hyperparameters, file paths and experimental settings
for QLoRA fine-tuning on AD-Align dataset.
"""
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for base model and QLoRA adapter"""
    base_model_name: str = "Qwen/Qwen-7B-Chat"
    load_in_4bit: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    compute_dtype: torch.dtype = torch.bfloat16


@dataclass
class DataConfig:
    """Configuration for dataset and preprocessing"""
    dataset_path: str = "./data/ad_align_dataset.csv"
    max_seq_length: int = 512
    val_split_ratio: float = 0.1
    prompt_template: str = """### Instruction:
{instruction}

### Input:
{input}

### Output:
"""


@dataclass
class TrainingConfig:
    """Configuration for training loop"""
    output_dir: str = "./output/qlora_ad_align_v1"
    num_train_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    seed: int = 42


@dataclass
class InferenceConfig:
    """Configuration for generation inference"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    adapter_path: str = "./output/qlora_ad_align_v1"
