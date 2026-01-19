import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaConfig:
    """
    LLaMA 모델 구성 클래스 (통합)
    """
    vocab_size: int = 32000
    hidden_dim: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    max_sequence_length: int = 4096
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Training/Inference 관련
    use_cache: bool = True
    max_batch_size: int = 8

    # FeedForward 관련
    scale_to_hidden_dim: int = 256
    ffn_dim_multiplier: Optional[float] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        
    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Hugging Face 모델 경로에서 config.json 로드
        """
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Huggingface config를 우리 config로 매핑
        return cls(
            vocab_size=config_dict.get("vocab_size", 32000),
            hidden_dim=config_dict.get("hidden_size", 4096),
            intermediate_size=config_dict.get("intermediate_size", 11008),
            num_hidden_layers=config_dict.get("num_hidden_layers", 32),
            num_attention_heads=config_dict.get("num_attention_heads", 32),
            num_key_value_heads=config_dict.get("num_key_value_heads", None),
            max_sequence_length=config_dict.get("max_position_embeddings", 4096),
            norm_eps=config_dict.get("rms_norm_eps", 1e-5),
            rope_theta=config_dict.get("rope_theta", 10000.0),
        )

    @classmethod
    def from_name(cls, model_name: str):
        """
        사전 정의된 모델 이름에서 설정 로드
        """
        llama_configs = {
            "llama-2-7b": {
                "vocab_size": 32000,
                "hidden_dim": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "max_sequence_length": 4096,
                "norm_eps": 1e-5,
            },
            "llama-3.1-8b": {
                "vocab_size": 128256,
                "hidden_dim": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_sequence_length": 8192,
                "norm_eps": 1e-5,
            },
        }

        model_name_lower = model_name.lower()
        if model_name_lower in llama_configs:
            return cls(**llama_configs[model_name_lower])
        else:
            raise ValueError(f"Unknown model name: {model_name}")