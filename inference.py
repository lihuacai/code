"""
Inference Deployment Script
Support both interactive chat mode and batch file inference mode.
Usage:
  - Interactive mode: python inference.py
  - Batch mode: python inference.py --input input.csv --output output.csv
"""
import argparse
import pandas as pd
from tqdm import tqdm

from main import QLoRAModel
from config import ModelConfig, InferenceConfig


def interactive_chat(qlora_model, infer_cfg: InferenceConfig) -> None:
    """Run interactive conversation mode in terminal."""
    print("=" * 60)
    print("QLoRA AD-Align Inference Mode")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        response = qlora_model.generate(
            prompt=user_input,
            max_new_tokens=infer_cfg.max_new_tokens,
            temperature=infer_cfg.temperature,
            top_p=infer_cfg.top_p,
            do_sample=infer_cfg.do_sample
        )
        print(f"Model: {response}")


def batch_inference(qlora_model, input_path: str, output_path: str, infer_cfg: InferenceConfig) -> None:
    """Run batch inference on CSV file with 'prompt' column."""
    df = pd.read_csv(input_path)
    if "prompt" not in df.columns:
        raise ValueError("Input CSV must contain a 'prompt' column")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        response = qlora_model.generate(
            prompt=row["prompt"],
            max_new_tokens=infer_cfg.max_new_tokens,
            temperature=infer_cfg.temperature,
            top_p=infer_cfg.top_p,
            do_sample=infer_cfg.do_sample
        )
        results.append({
            "prompt": row["prompt"],
            "response": response
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA Model Inference")
    parser.add_argument("--input", type=str, help="Path to input CSV file for batch inference")
    parser.add_argument("--output", type=str, help="Path to save batch inference results")
    args = parser.parse_args()

    # Load configurations
    model_cfg = ModelConfig()
    infer_cfg = InferenceConfig()

    # Initialize model
    print("Loading model...")
    qlora_model = QLoRAModel(
        base_model_name=model_cfg.base_model_name,
        load_in_4bit=model_cfg.load_in_4bit,
        is_train_mode=False
    )

    # Load trained adapter
    qlora_model.load_adapter(infer_cfg.adapter_path)

    # Run inference mode
    if args.input and args.output:
        batch_inference(qlora_model, args.input, args.output, infer_cfg)
    else:
        interactive_chat(qlora_model, infer_cfg)


if __name__ == "__main__":
    main()
