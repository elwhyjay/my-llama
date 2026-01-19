import json
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import LLamaTransformer
from model.tokenizer import Tokenizer
from config.model_config import LlamaConfig

Role = Literal["system", "user", "assistant"]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

class Message(TypedDict):
    role: Role
    content: str

class CompletionPredict(TypedDict, total=False):
    generation: str
    tokens: List[str]
    logprobs: List[float]

class ChatPredict(TypedDict, total=False):
    generation: Message
    tokens: List[str]
    logprobs: List[float]

Dialog = List[Message]

class Llama:
    def __init__(self, model: LLamaTransformer, tokenizer: Tokenizer, config: LlamaConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @staticmethod
    def build(
        model_path: str,
        tokenizer_path: Optional[str] = None,
        max_batch_size: int = 1,
        max_seq_len: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> "Llama":
        """
        Llama 모델 빌드

        Args:
            model_path: Huggingface 모델 경로 또는 체크포인트 경로
            tokenizer_path: 토크나이저 경로 (None이면 model_path에서 찾음)
            max_batch_size: 최대 배치 크기
            max_seq_len: 최대 시퀀스 길이
            device: 디바이스 (cuda/cpu)
            dtype: 모델 데이터 타입 (float16, bfloat16, float32)

        Returns:
            Llama 인스턴스
        """
        # 1. Config 로드
        print(f"Loading config from {model_path}")
        config = LlamaConfig.from_pretrained(model_path)
        config.max_batch_size = max_batch_size
        config.max_sequence_length = max_seq_len

        # 2. Tokenizer 로드
        if tokenizer_path is None:
            tokenizer_path = os.path.join(model_path, "tokenizer.model")

        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer(tokenizer_path)

        # 3. 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. 모델 초기화 (CPU에서 초기화하여 GPU 메모리 절약)
        print(f"Initializing model on CPU with config: {config}")
        model = LLamaTransformer(config)
        model.eval()

        # 5. 가중치 로드 (CPU로 로드)
        print(f"Loading weights from {model_path} with dtype {dtype}")
        Llama._load_weights(model, model_path, device='cpu', dtype=dtype)

        # 6. GPU로 이동 (한 번에 이동)
        print(f"Moving model to {device}")
        model = model.to(device).to(dtype)

        print(f"Model loaded successfully on {device} with dtype {dtype}")
        return Llama(model, tokenizer, config)

    @staticmethod
    def _load_weights(model: LLamaTransformer, model_path: str, device: str, dtype: torch.dtype = torch.float16):
        """
        Huggingface 체크포인트에서 가중치 로드 (CPU에서)
        """
        # pytorch_model.bin 또는 model.safetensors 찾기
        bin_files = []

        # 단일 파일 체크
        single_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(single_file):
            bin_files = [single_file]
        else:
            # 샤딩된 파일들 찾기
            import glob
            pattern = os.path.join(model_path, "pytorch_model-*.bin")
            bin_files = sorted(glob.glob(pattern))

        if not bin_files:
            raise FileNotFoundError(f"No model weights found in {model_path}")

        print(f"Found {len(bin_files)} weight file(s)")

        # 가중치 로드 및 매핑
        state_dict = {}
        for bin_file in bin_files:
            print(f"Loading {bin_file}")
            # CPU에 로드
            weights = torch.load(bin_file, map_location='cpu', weights_only=True)
            # dtype 변환 (in-place로 메모리 절약)
            for key in list(weights.keys()):
                if weights[key].dtype != dtype:
                    weights[key] = weights[key].to(dtype)
            state_dict.update(weights)
            del weights  # 메모리 해제

        # Huggingface 키를 우리 모델 키로 매핑
        print("Mapping Huggingface weights to model structure")
        mapped_state_dict = Llama._map_huggingface_weights(state_dict)
        del state_dict  # 원본 state_dict 메모리 해제

        # 모델에 로드 (CPU에 있는 model에 로드)
        model.load_state_dict(mapped_state_dict, strict=False)
        del mapped_state_dict  # 메모리 해제
        print("Weights loaded successfully on CPU")

    @staticmethod
    def _map_huggingface_weights(hf_state_dict: dict) -> dict:
        """
        Huggingface 가중치 키를 우리 모델의 키로 매핑

        Huggingface Llama 구조:
        - model.embed_tokens.weight → token_embedding.embedding.weight
        - model.layers.{i}.self_attn.q_proj.weight → layers.{i}.attention.q_proj.weight
        - model.layers.{i}.self_attn.k_proj.weight → layers.{i}.attention.k_proj.weight
        - model.layers.{i}.self_attn.v_proj.weight → layers.{i}.attention.v_proj.weight
        - model.layers.{i}.self_attn.o_proj.weight → layers.{i}.attention.o_proj.weight
        - model.layers.{i}.mlp.gate_proj.weight → layers.{i}.ffn.ffn.w1.weight
        - model.layers.{i}.mlp.up_proj.weight → layers.{i}.ffn.ffn.w2.weight
        - model.layers.{i}.mlp.down_proj.weight → layers.{i}.ffn.ffn.w3.weight
        - model.layers.{i}.input_layernorm.weight → layers.{i}.norm.weight
        - model.layers.{i}.post_attention_layernorm.weight → layers.{i}.ffn_norm.weight
        - model.norm.weight → norm.weight
        - lm_head.weight → output.weight
        """
        mapped_dict = {}

        for key, value in hf_state_dict.items():
            new_key = key

            # Embedding
            if key == "model.embed_tokens.weight":
                new_key = "token_embedding.embedding.weight"

            # Transformer layers
            elif key.startswith("model.layers."):
                new_key = key.replace("model.layers.", "layers.")
                new_key = new_key.replace("self_attn.", "attention.")
                new_key = new_key.replace("mlp.gate_proj.", "ffn.ffn.w1.")
                new_key = new_key.replace("mlp.up_proj.", "ffn.ffn.w2.")
                new_key = new_key.replace("mlp.down_proj.", "ffn.ffn.w3.")
                new_key = new_key.replace("input_layernorm.", "norm.")
                new_key = new_key.replace("post_attention_layernorm.", "ffn_norm.")

            # Final norm
            elif key == "model.norm.weight":
                new_key = "norm.weight"

            # Output projection
            elif key == "lm_head.weight":
                new_key = "output.weight"

            mapped_dict[new_key] = value

        return mapped_dict

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        토큰 생성

        Args:
            prompt_tokens: 배치별 프롬프트 토큰 리스트
            max_gen_len: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            top_p: nucleus sampling 파라미터
            echo: 프롬프트를 결과에 포함할지 여부

        Returns:
            생성된 토큰 리스트와 로그 확률 (옵션)
        """
        batch_size = len(prompt_tokens)
        assert batch_size <= self.config.max_batch_size, f"Batch size {batch_size} exceeds max {self.config.max_batch_size}"

        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.max_sequence_length, f"Prompt length {max_prompt_len} exceeds max {self.config.max_sequence_length}"

        total_len = min(self.config.max_sequence_length, max_gen_len + max_prompt_len)

        # 패딩된 토큰 텐서 생성
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.model.token_embedding.embedding.weight.device)

        for i, t in enumerate(prompt_tokens):
            tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)

        eos_reached = torch.tensor([False] * batch_size, device=tokens.device)
        prompt_tokens_mask = tokens != pad_id
        input_text_mask = prompt_tokens_mask.clone()

        # 첫 forward pass: 전체 프롬프트 처리하여 KV 캐시 채우기
        logits = self.model(tokens[:, :max_prompt_len], start_pos=0, use_cache=True)

        # 토큰 생성 루프
        for cur_pos in range(max_prompt_len, total_len):
            # 다음 토큰 예측
            if temperature > 0:
                # Temperature scaling
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            # 이미 생성이 끝난 시퀀스는 pad 토큰으로 대체
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # EOS 체크
            eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

            # 다음 토큰으로 forward pass (KV 캐시 활용)
            if cur_pos < total_len - 1:
                logits = self.model(tokens[:, cur_pos:cur_pos+1], start_pos=cur_pos, use_cache=True)

        # 결과 추출
        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # EOS까지 또는 패딩 전까지
            if echo:
                start = 0
            else:
                start = len(prompt_tokens[i])

            toks = toks[start:total_len]

            # EOS나 패딩 제거
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]

            out_tokens.append(toks)

        return out_tokens, None

    def _sample_top_p(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Nucleus sampling (top-p sampling)
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        echo: bool = False,
    ) -> List[CompletionPredict]:
        """
        텍스트 완성

        Args:
            prompts: 프롬프트 텍스트 리스트
            temperature: 샘플링 온도
            top_p: nucleus sampling 파라미터
            max_gen_len: 생성할 최대 토큰 수 (None이면 config에서 가져옴)
            echo: 프롬프트를 결과에 포함할지 여부

        Returns:
            완성된 텍스트 결과 리스트
        """
        if max_gen_len is None:
            max_gen_len = self.config.max_sequence_length - 1

        # 프롬프트 토큰화
        prompt_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

        # 생성
        generation_tokens, _ = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )

        # 디코딩
        results = []
        for tokens in generation_tokens:
            text = self.tokenizer.decode(tokens)
            results.append({"generation": text})

        return results
