"""
Llama 모델 Inference 테스트 스크립트
"""
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from model.generation import Llama


def download_model(model_name: str = "meta-llama/Llama-2-7b-hf", cache_dir: str = "./models"):
    """
    Huggingface에서 모델 다운로드

    Args:
        model_name: Huggingface 모델 이름
        cache_dir: 다운로드할 디렉토리

    Returns:
        다운로드된 모델의 로컬 경로
    """
    print(f"Downloading {model_name} from Huggingface...")
    print("Note: This requires Huggingface authentication for Llama models.")
    print("Please run: huggingface-cli login")
    print()

    try:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.bin", "*.model", "*.safetensors"],
        )
        print(f"Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nPlease make sure you:")
        print("1. Have accepted the Llama license on Huggingface")
        print("2. Have logged in with: huggingface-cli login")
        raise


def test_inference(
    model_path: str,
    prompts: list[str],
    max_gen_len: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Inference 테스트

    Args:
        model_path: 모델 경로
        prompts: 테스트할 프롬프트 리스트
        max_gen_len: 생성할 최대 토큰 수
        temperature: 샘플링 온도
        top_p: nucleus sampling 파라미터
    """
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)

    # 모델 로드 (메모리 절약을 위해 FP16 사용)
    llama = Llama.build(
        model_path=model_path,
        max_seq_len=256,  # 메모리 절약을 위해 줄임
        max_batch_size=1,  # 배치 크기 1로 제한
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,  # FP16 사용
    )

    print("\n" + "="*80)
    print("Running inference...")
    print("="*80)

    # # Inference 실행
    # results = llama.text_completion(
    #     prompts=prompts,
    #     temperature=temperature,
    #     top_p=top_p,
    #     max_gen_len=max_gen_len,
    #     echo=False,
    # )
    # Inference 실행 (한 번에 하나씩 처리)
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}...")
        result = llama.text_completion(
            prompts=[prompt],  # 한 번에 하나씩
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            echo=False,
        )
        results.extend(result)

        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 결과 출력
    print("\n" + "="*80)
    print("Results:")
    print("="*80)

    for i, (prompt, result) in enumerate(zip(prompts, results)):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Generated: {result['generation']}")
        print()


def main():
    # 테스트 프롬프트
    prompts = [
        "Once upon a time",
        "The capital of Korea is",
    ]

    # 모델 경로 (환경 변수 또는 기본값)
    model_path = os.environ.get("LLAMA_MODEL_PATH")

    if model_path is None:
        print("LLAMA_MODEL_PATH not set. Attempting to download Llama-2-7b-hf...")
        print("")

        # 모델 다운로드
        try:
            model_path = download_model(
                model_name="meta-llama/Llama-2-7b-hf",
                cache_dir="./models"
            )
        except Exception as e:
            print(f"\nFailed to download model: {e}")
            print("\nAlternatively, you can:")
            print("1. Download the model manually")
            print("2. Set LLAMA_MODEL_PATH environment variable to the model directory")
            sys.exit(1)

    # Inference 테스트 (메모리 절약을 위해 짧은 생성)
    test_inference(
        model_path=model_path,
        prompts=prompts,
        max_gen_len=32,  # 짧게 생성
        temperature=0.7,
        top_p=0.9,
    )

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
