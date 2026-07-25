"""
QLoRA Adapter Module for Large Language Models
Corresponding to Figure 2 & Figure 3 in the paper: 4-bit NF4 quantized base model + Low-Rank Adapter (LoRA)
Implements lightweight instruction fine-tuning with only ~0.1% of parameters trainable
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)


class QLoRAModel:
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen-7B-Chat",
        r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: list = None,
        load_in_4bit: bool = True,
        is_train_mode: bool = True
    ):
        """
        Initialize QLoRA model
        :param base_model_name: Local path of the base large model or HuggingFace repository name
        :param r: Rank of LoRA matrices, default set to 64 in the paper
        :param lora_alpha: LoRA scaling factor, usually set to 2 times of r
        :param lora_dropout: Dropout probability for LoRA layers
        :param target_modules: Target modules to inject LoRA, default to attention Q/K/V/O projection layers
        :param load_in_4bit: Whether to load the base model with 4-bit quantization
        :param is_train_mode: Whether to enable training mode; LoRA will be injected and gradients enabled in training mode
        """
        self.base_model_name = base_model_name

        # Device check: 4-bit quantization only supports CUDA environment
        if load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit quantization requires a CUDA environment. Please install the GPU version of PyTorch and ensure a valid GPU is available. "
                "Set load_in_4bit=False for CPU environments."
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ========== 1. Load Tokenizer ==========
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        # Compatible with models without default pad_token (e.g. Qwen) to avoid errors during generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ========== 2. 4-bit NF4 Quantization Configuration (corresponds to the quantization branch on the left of Figure 2) ==========
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",          # NormalFloat 4 quantization, better distribution adaptability than FP4
            bnb_4bit_compute_dtype=torch.bfloat16,  # Dequantize to bf16 during forward computation
            bnb_4bit_use_double_quant=True,    # Double quantization: further compress quantization constants to reduce VRAM usage
        )

        # ========== 3. Load Base Large Model ==========
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # ========== 4. Training Mode: Inject LoRA Adapter ==========
        if is_train_mode:
            # Preprocessing for k-bit training: freeze quantized weights and enable gradients for input layers
            base_model = prepare_model_for_kbit_training(base_model)

            # LoRA configuration (corresponds to the right side of Figure 2: low-rank matrices A and B)
            if target_modules is None:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            lora_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Inject adapter: weight update is equivalent to ΔW = B * A
            self.model = get_peft_model(base_model, lora_config)

            # Print trainable parameter count (typical QLoRA value: ~0.1% of total parameters)
            print("=" * 60)
            print("QLoRA model initialization completed. Trainable parameters statistics:")
            self.model.print_trainable_parameters()
            print("=" * 60)

        # ========== 5. Inference Mode: Load base model only, LoRA weights can be loaded later ==========
        else:
            self.model = base_model
            self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Model generation inference
        :param prompt: Input prompt text
        :param max_new_tokens: Maximum number of tokens to generate
        :param temperature: Sampling temperature; higher value means more randomness
        :param top_p: Nucleus sampling probability threshold
        :param do_sample: Whether to enable stochastic sampling
        :return: Complete generated text (including input prompt, consistent with original code logic)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_adapter(self, save_dir: str) -> None:
        """
        Save LoRA adapter weights (only saves trainable low-rank matrices, resulting in very small file size)
        :param save_dir: Directory to save weights
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f" LoRA adapter saved to: {os.path.abspath(save_dir)}")

    def load_adapter(self, adapter_dir: str, adapter_name: str = "default") -> None:
        """
        Load a trained LoRA adapter for inference
        :param adapter_dir: Path to the LoRA weight directory
        :param adapter_name: Adapter name, supports switching between multiple adapters
        """
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_dir,
            adapter_name=adapter_name
        )
        self.model.eval()
        print(f" LoRA adapter [{adapter_name}] loaded: {os.path.abspath(adapter_dir)}")


if __name__ == "__main__":
    # ========== Quick Test Example ==========
    print("Initializing QLoRA model (training mode)...")
    
    try:
        # Initialize model
        qlora_llm = QLoRAModel(
            base_model_name="Qwen/Qwen-7B-Chat",
            r=64,
            is_train_mode=True
        )

        # Test generation
        test_prompt = "Please introduce Alzheimer's disease in one sentence."
        print(f"\nInput prompt: {test_prompt}")
        
        result = qlora_llm.generate(test_prompt, max_new_tokens=50)
        print(f"Generation result: {result}")

        # Save adapter weights after training
        # qlora_llm.save_adapter("./output/qlora_adapter_v1")

    except Exception as e:
        print(f"Runtime error: {e}")
        print("Hint: Please ensure dependencies are installed and a valid CUDA environment is available.")
