class LlamaConfig:
    """
    LLaMA 모델 구성 클래스
    """
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,  # 피드포워드 레이어의 중간 크기
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA의 키/값 헤드 수
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_theta=10000.0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        """
        Hugging Face 모델 이름에서 구성 로드 (예시)
        실제 구현에서는 Hugging Face의 구성 파일에서 로드해야 함
        """
        llama_configs = {
            "Llama-3.1-8B": {
                "vocab_size": 128256,  
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-5,
            },
            "Llama-3.1-70B": {
                "vocab_size": 128256,  
                "hidden_size": 8192,
                "intermediate_size": 28672,
                "num_hidden_layers": 80,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-5,
            },
        }
        
        if model_name_or_path in llama_configs:
            return cls(**llama_configs[model_name_or_path])
        else:
            raise ValueError(f"Unknown model name: {model_name_or_path}")