"""
QLoRA 大语言模型适配器模块
对应论文 Fig 2 & Fig 3：4-bit NF4 量化基座 + 低秩适配器 (LoRA)
实现轻量化指令微调，仅训练约 0.1% 的参数
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
        初始化 QLoRA 模型
        :param base_model_name: 基座大模型本地路径或 HuggingFace 仓库名
        :param r: LoRA 矩阵秩 (Rank)，论文默认设置为 64
        :param lora_alpha: LoRA 缩放系数，通常取 r 的 2 倍
        :param lora_dropout: LoRA 层的 dropout 概率
        :param target_modules: 注入 LoRA 的目标模块，默认为注意力 Q/K/V/O 投影层
        :param load_in_4bit: 是否启用 4-bit 量化加载基座
        :param is_train_mode: 是否为训练模式，训练模式会注入 LoRA 并启用梯度
        """
        self.base_model_name = base_model_name

        # 设备校验：4bit 量化仅支持 CUDA 环境
        if load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit 量化依赖 CUDA 环境，请安装 GPU 版本 PyTorch 并确保显卡可用。"
                "CPU 环境请设置 load_in_4bit=False。"
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ========== 1. 加载分词器 ==========
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        # 兼容 Qwen 等无默认 pad_token 的模型，避免生成时报错
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ========== 2. 4-bit NF4 量化配置 (对应 Fig 2 左侧量化分支) ==========
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",          # NormalFloat 4 量化，分布适配性优于 FP4
            bnb_4bit_compute_dtype=torch.bfloat16,  # 前向计算时反量化为 bf16
            bnb_4bit_use_double_quant=True,    # 双重量化，对量化常数再次压缩，进一步降低显存
        )

        # ========== 3. 加载基座大模型 ==========
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # ========== 4. 训练模式：注入 LoRA 适配器 ==========
        if is_train_mode:
            # 为 k-bit 训练预处理：冻结量化权重、开启输入层梯度
            base_model = prepare_model_for_kbit_training(base_model)

            # LoRA 配置 (对应 Fig 2 右侧：低秩矩阵 A、B)
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

            # 注入适配器：权重更新等效为 ΔW = B * A
            self.model = get_peft_model(base_model, lora_config)

            # 打印可训练参数量（QLoRA 典型值：约 0.1% 总参数）
            print("=" * 60)
            print("QLoRA 模型初始化完成，可训练参数统计：")
            self.model.print_trainable_parameters()
            print("=" * 60)

        # ========== 5. 推理模式：仅加载基座，后续可加载 LoRA 权重 ==========
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
        模型生成推理
        :param prompt: 输入提示文本
        :param max_new_tokens: 最大生成 token 数
        :param temperature: 采样温度，越高随机性越强
        :param top_p: 核采样概率阈值
        :param do_sample: 是否启用随机采样
        :return: 完整的生成文本（含输入 prompt，与原代码逻辑一致）
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
        保存 LoRA 适配器权重（仅保存可训练的低秩矩阵，文件体积极小）
        :param save_dir: 权重保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"✅ LoRA 适配器已保存至: {os.path.abspath(save_dir)}")

    def load_adapter(self, adapter_dir: str, adapter_name: str = "default") -> None:
        """
        加载已训练的 LoRA 适配器，用于推理阶段
        :param adapter_dir: LoRA 权重目录路径
        :param adapter_name: 适配器命名，支持多适配器切换
        """
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_dir,
            adapter_name=adapter_name
        )
        self.model.eval()
        print(f"✅ 已加载 LoRA 适配器 [{adapter_name}]: {os.path.abspath(adapter_dir)}")


if __name__ == "__main__":
    # ========== 快速测试示例 ==========
    print("正在初始化 QLoRA 模型（训练模式）...")
    
    try:
        # 初始化模型
        qlora_llm = QLoRAModel(
            base_model_name="Qwen/Qwen-7B-Chat",
            r=64,
            is_train_mode=True
        )

        # 测试生成
        test_prompt = "请用一句话介绍阿尔茨海默病。"
        print(f"\n输入提示：{test_prompt}")
        
        result = qlora_llm.generate(test_prompt, max_new_tokens=50)
        print(f"生成结果：{result}")

        # 训练完成后保存适配器权重
        # qlora_llm.save_adapter("./output/qlora_adapter_v1")

    except Exception as e:
        print(f"运行出错：{e}")
        print("提示：请确保已安装依赖且拥有可用的 CUDA 环境。")