"""
Data Processing Module
Dataset loading, prompt formatting and tokenization pipeline for AD-Align dataset.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple
from transformers import PreTrainedTokenizer

from config import DataConfig


class ADAlignDataset(Dataset):
    """
    PyTorch Dataset for AD-Align instruction tuning dataset.
    Expects CSV file with columns: instruction, input, output
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        prompt_template: str = None
    ):
        """
        Initialize AD-Align dataset.
        :param data_path: Path to CSV dataset file
        :param tokenizer: Model tokenizer instance
        :param max_seq_length: Maximum sequence length for tokenization
        :param prompt_template: Template for instruction formatting
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt_template = prompt_template or DataConfig.prompt_template
        
        # Load and validate dataset
        self.data = pd.read_csv(data_path)
        required_cols = ["instruction", "input", "output"]
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(
                f"Dataset must contain columns: {required_cols}. "
                f"Found columns: {list(self.data.columns)}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        
        # Format prompt with instruction-input-output structure
        full_prompt = self.prompt_template.format(
            instruction=row["instruction"],
            input=row["input"]
        )
        full_text = full_prompt + str(row["output"])

        # Tokenize full text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # For causal LM training: labels = input_ids, ignore padding tokens
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    data_config: DataConfig,
    batch_size: int = 8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders with random split.
    :param tokenizer: Model tokenizer instance
    :param data_config: Data configuration object
    :param batch_size: Batch size for dataloaders
    :param seed: Random seed for train/val split
    :return: Tuple of (train_dataloader, val_dataloader)
    """
    full_dataset = ADAlignDataset(
        data_path=data_config.dataset_path,
        tokenizer=tokenizer,
        max_seq_length=data_config.max_seq_length,
        prompt_template=data_config.prompt_template
    )

    # Calculate split sizes
    val_size = int(len(full_dataset) * data_config.val_split_ratio)
    train_size = len(full_dataset) - val_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader
