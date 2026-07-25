"""
Evaluation Script
Calculate perplexity on validation set and run batch generation for qualitative analysis.
Usage: python evaluate.py
"""
import os
import math
import pandas as pd
import torch
from tqdm import tqdm

from main import QLoRAModel
from config import ModelConfig, DataConfig, TrainingConfig, InferenceConfig
from data_utils import create_dataloaders
from utils import setup_logger


def calculate_perplexity(qlora_model, val_loader) -> float:
    """
    Calculate perplexity (PPL) on the validation dataset.
    :param qlora_model: QLoRAModel instance
    :param val_loader: Validation dataloader
    :return: Perplexity score
    """
    model = qlora_model.model
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calculating perplexity"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            
            # Count non-padding tokens
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def batch_generation(
    qlora_model,
    input_file: str,
    output_file: str,
    infer_cfg: InferenceConfig
) -> None:
    """
    Run batch inference on input prompts and save results.
    :param qlora_model: QLoRAModel instance
    :param input_file: Path to CSV file with prompts
    :param output_file: Path to save generation results
    :param infer_cfg: Inference configuration
    """
    df = pd.read_csv(input_file)
    if "prompt" not in df.columns:
        raise ValueError("Input CSV must contain a 'prompt' column")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
        prompt = row["prompt"]
        response = qlora_model.generate(
            prompt=prompt,
            max_new_tokens=infer_cfg.max_new_tokens,
            temperature=infer_cfg.temperature,
            top_p=infer_cfg.top_p,
            do_sample=infer_cfg.do_sample
        )
        results.append({
            "prompt": prompt,
            "response": response
        })

    output_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Generation results saved to {output_file}")


def main():
    # Load configurations
    model_cfg = ModelConfig()
    data_cfg = DataConfig()
    train_cfg = TrainingConfig()
    infer_cfg = InferenceConfig()

    logger = setup_logger(name="qlora_eval")
    logger.info("Starting model evaluation...")

    # Initialize model and load trained adapter
    logger.info(f"Loading base model: {model_cfg.base_model_name}")
    qlora_model = QLoRAModel(
        base_model_name=model_cfg.base_model_name,
        load_in_4bit=model_cfg.load_in_4bit,
        is_train_mode=False
    )
    
    logger.info(f"Loading LoRA adapter from: {infer_cfg.adapter_path}")
    qlora_model.load_adapter(infer_cfg.adapter_path)

    # 1. Calculate perplexity
    _, val_loader = create_dataloaders(
        tokenizer=qlora_model.tokenizer,
        data_config=data_cfg,
        batch_size=train_cfg.batch_size,
        seed=train_cfg.seed
    )
    
    ppl = calculate_perplexity(qlora_model, val_loader)
    logger.info(f"Validation Perplexity: {ppl:.2f}")

    # 2. Run batch generation (optional)
    input_prompt_file = "./data/eval_prompts.csv"
    output_result_file = "./output/evaluation_results.csv"
    if os.path.exists(input_prompt_file):
        logger.info("Running batch generation...")
        batch_generation(qlora_model, input_prompt_file, output_result_file, infer_cfg)
    else:
        logger.info(f"Eval prompt file {input_prompt_file} not found. Skipping batch generation.")

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
