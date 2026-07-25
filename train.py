"""
Training Script
Full QLoRA fine-tuning pipeline for AD-Align dataset with validation,
logging, checkpoint saving and learning rate scheduling.
Usage: python train.py
"""
import os
import math
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from main import QLoRAModel
from config import ModelConfig, DataConfig, TrainingConfig
from data_utils import create_dataloaders
from utils import set_seed, setup_logger, get_gpu_memory_usage


def main():
    # Load configurations
    model_cfg = ModelConfig()
    data_cfg = DataConfig()
    train_cfg = TrainingConfig()

    # Initialize
    set_seed(train_cfg.seed)
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(train_cfg.output_dir, "training.log"))
    logger.info("=" * 70)
    logger.info("Starting QLoRA fine-tuning for AD-Align dataset")
    logger.info(f"Base model: {model_cfg.base_model_name}")
    logger.info(f"Output directory: {train_cfg.output_dir}")
    logger.info("=" * 70)

    # Initialize QLoRA model in training mode
    logger.info("Initializing QLoRA model...")
    qlora_model = QLoRAModel(
        base_model_name=model_cfg.base_model_name,
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=model_cfg.target_modules,
        load_in_4bit=model_cfg.load_in_4bit,
        is_train_mode=True
    )
    model = qlora_model.model
    tokenizer = qlora_model.tokenizer
    logger.info(f"GPU Memory after model loading: {get_gpu_memory_usage()}")

    # Create dataloaders
    logger.info("Loading AD-Align dataset...")
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        data_config=data_cfg,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed
    )
    logger.info(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    # Prepare optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )

    total_steps = len(train_loader) * train_cfg.num_train_epochs // train_cfg.gradient_accumulation_steps
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    logger.info("Starting training...")

    for epoch in range(train_cfg.num_train_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / train_cfg.gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item() * train_cfg.gradient_accumulation_steps

            # Optimizer step
            if (step + 1) % train_cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % train_cfg.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    logger.info(
                        f"Epoch {epoch+1}/{train_cfg.num_train_epochs} | "
                        f"Step {global_step} | "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )

                # Validation
                if global_step % train_cfg.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, logger)
                    logger.info(f"Validation Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}")
                    
                    # Save best checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_save_path = os.path.join(train_cfg.output_dir, "best_adapter")
                        qlora_model.save_adapter(best_save_path)
                        logger.info(f"New best model saved to {best_save_path}")
                    
                    model.train()

                # Save checkpoint
                if global_step % train_cfg.save_steps == 0:
                    checkpoint_path = os.path.join(train_cfg.output_dir, f"checkpoint_step_{global_step}")
                    qlora_model.save_adapter(checkpoint_path)

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed. Average training loss: {avg_epoch_loss:.4f}")

    # Save final adapter
    final_save_path = os.path.join(train_cfg.output_dir, "final_adapter")
    qlora_model.save_adapter(final_save_path)
    logger.info("Training completed. Final adapter saved.")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Peak GPU memory: {get_gpu_memory_usage()}")


def evaluate(model, val_loader, logger) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


if __name__ == "__main__":
    main()
