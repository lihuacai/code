# QLoRA Large Language Model Adapter: Lightweight Fine-tuning for Alzheimer's Disease Research

## Description

This repository implements the QLoRA (Quantized Low-Rank Adaptation) algorithm for lightweight instruction fine-tuning of large language models (LLMs). It adopts 4-bit NormalFloat (NF4) quantization for the base model and injects low-rank adapter (LoRA) modules into transformer attention layers, achieving effective domain adaptation with only ~0.1% of trainable parameters.

This code is designed for Alzheimer's disease (AD) related natural language processing tasks and is fully compatible with the AD-Align dataset. It supports both training and inference modes, enabling efficient model adaptation on consumer-grade GPUs with minimal memory overhead.

## Dataset Information

- **Dataset Name**: AD-Align Dataset
- **Purpose**: A domain-specific curated dataset for Alzheimer's disease research, used for supervised instruction fine-tuning to improve LLM performance on AD-related question answering, clinical text analysis, and knowledge reasoning tasks.
- **Format**: Provided in structured CSV/JSON format, containing instruction-input-output triples for fine-tuning.
- **Access**: The AD-Align dataset is available at \[DOI/URL: fill in your dataset DOI or repository link here\]. Please download the dataset and place it in the `./data/` directory before running fine-tuning.

## Code Information

- **Core File**: `main.py`
- **Main Structure**:
  - `QLoRAModel` class: Encapsulates the full QLoRA model pipeline, including base model loading, quantization configuration, LoRA injection, text generation, and adapter persistence.
  - Training mode: Freezes the quantized base model and enables gradient computation only for LoRA parameters.
  - Inference mode: Loads the base model and supports loading pre-trained LoRA adapters for downstream task deployment.
- **Key Features**:
  - 4-bit NF4 quantization with double quantization for maximum VRAM reduction
  - Configurable LoRA hyperparameters (rank, scaling factor, dropout rate, target modules)
  - Compatible with all causal LLM architectures supported by HuggingFace Transformers
  - Built-in adapter save/load functionality for multi-scenario deployment

## Requirements

### Environment Prerequisites

- Python >= 3.8
- CUDA >= 11.7 (mandatory for 4-bit quantization; 4-bit training is not supported on CPU)
- NVIDIA GPU with at least 8 GB VRAM (for 7B-parameter base models)

### Dependencies

Install required packages via `pip`:

```
pip install torch>=2.0.0 transformers>=4.30.0 peft>=0.4.0 bitsandbytes>=0.40.0 accelerate>=0.20.0
```

## Usage Instructions

### 1. Quick Test

Run the built-in example to verify environment configuration and model initialization:

```
python main.py
```

By default, this script initializes a QLoRA model based on `Qwen/Qwen-7B-Chat` in training mode and runs a sample generation test.

### 2. Initialize Model for Fine-tuning

Import the `QLoRAModel` class and initialize it in training mode, then integrate it with your custom training loop and data loader:

```
from main import QLoRAModel

# Initialize QLoRA model in training mode
qlora_model = QLoRAModel(
    base_model_name="Qwen/Qwen-7B-Chat",  # Replace with your base model path
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    load_in_4bit=True,
    is_train_mode=True
)

# Access the underlying model and tokenizer for training
model = qlora_model.model
tokenizer = qlora_model.tokenizer

# Integrate with your training loop, data loader and optimizer
# ...
```

### 3. Run Inference with Trained Adapter

Initialize the model in inference mode and load a pre-trained LoRA adapter for text generation:

```
from main import QLoRAModel

# Initialize base model in inference mode
qlora_model = QLoRAModel(
    base_model_name="Qwen/Qwen-7B-Chat",
    load_in_4bit=True,
    is_train_mode=False
)

# Load pre-trained LoRA adapter weights
qlora_model.load_adapter("./output/qlora_adapter_v1")

# Generate text
prompt = "Please introduce Alzheimer's disease in one sentence."
output = qlora_model.generate(
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)
print(output)
```

### 4. Save Trained Adapter

After fine-tuning, save the LoRA adapter weights (only trainable parameters, extremely small file size):

```
qlora_model.save_adapter("./output/qlora_adapter_v1")
```

## Methodology

This implementation follows the QLoRA framework with the following technical design, corresponding to Figure 2 and Figure 3 in the associated manuscript:

1. **4-bit NF4 Quantization**: The base LLM is loaded in 4-bit NormalFloat (NF4) format, which maintains model performance while drastically reducing memory footprint. Double quantization is applied to further compress quantization constants.
2. **Low-Rank Adaptation**: Trainable low-rank matrices are injected into the query, key, value, and output projection layers of the transformer attention module. Weight updates are computed as ΔW = BA, eliminating the need for full-parameter fine-tuning.
3. **Mixed Precision Computation**: Forward and backward computations are executed in bfloat16 precision, with on-the-fly dequantization during computation.
4. **Parameter Efficiency**: Only approximately 0.1% of total model parameters are trainable, enabling efficient domain adaptation on consumer-grade GPUs.

## Citations

If you use this code or the AD-Align dataset in your research, please cite the following works:

1. QLoRA original paper:

```
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```

......

## License & Contribution Guidelines

- **License**: This project is released under the Apache 2.0 License. See the `LICENSE` file for full terms.
- **Contributions**: Bug reports and pull requests are welcome. For major modifications, please open an issue first to discuss the proposed changes.