"""
속도 벤치마킹 스크립트

torch profiler를 사용하여 inference 성능을 측정합니다:
- 전체 생성 속도 (tokens/sec)
- 프롬프트 처리 시간 (prefill phase)
- 토큰 생성 시간 (decoding phase)
- 레이어별 시간 분석
- 병목 지점 파악
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Dict
import json

import torch
import torch.profiler as profiler
from huggingface_hub import snapshot_download

import sys
sys.path.append(str(Path(__file__).parent.parent))

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


class SpeedBenchmark:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 512,
    ):
        print("=" * 80)
        print("Initializing Speed Benchmark")
        print("=" * 80)

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # 모델 로드
        print(f"\nLoading model from: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")

        self.llama = Llama.build(
            model_path=model_path,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        print("\nModel loaded successfully!")

    def warmup(self, num_runs: int = 3):
        """워밍업: 첫 실행 시 컴파일/초기화 비용 제거"""
        print("\n" + "=" * 80)
        print(f"Warming up ({num_runs} runs)...")
        print("=" * 80)

        warmup_prompt = "Hello, how are you?"

        for i in range(num_runs):
            _ = self.llama.text_completion(
                prompts=[warmup_prompt],
                max_gen_len=10,
                temperature=0.0,
                echo=False,
            )
            print(f"Warmup run {i+1}/{num_runs} completed")

        # GPU 동기화
        if self.device == "cuda":
            torch.cuda.synchronize()

        print("Warmup completed!\n")

    def benchmark_simple(
        self,
        prompts: List[str],
        max_gen_len: int = 100,
        temperature: float = 0.0,
    ) -> Dict:
        """간단한 벤치마크: 전체 시간과 tokens/sec 측정"""

        # GPU 동기화
        if self.device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        results = self.llama.text_completion(
            prompts=prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            echo=False,
        )

        # GPU 동기화
        if self.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # 생성된 토큰 수 계산
        total_tokens = sum(len(self.llama.tokenizer.encode(r["generation"], bos=False, eos=False)) for r in results)
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0

        return {
            "elapsed_time": elapsed_time,
            "total_tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "prompts": prompts,
            "results": [r["generation"] for r in results],
        }

    def benchmark_with_profiler(
        self,
        prompt: str,
        max_gen_len: int = 100,
        temperature: float = 0.0,
        output_dir: str = "./benchmark_results",
    ) -> Dict:
        """torch profiler를 사용한 상세 벤치마크"""

        print("\n" + "=" * 80)
        print("Running detailed benchmark with torch profiler...")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Profiler 설정
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA if self.device == "cuda" else profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            with profiler.record_function("text_completion"):
                result = self.llama.text_completion(
                    prompts=[prompt],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    echo=False,
                )

        # GPU 동기화
        if self.device == "cuda":
            torch.cuda.synchronize()

        # 결과 저장
        trace_file = output_path / "profiler_trace.json"
        prof.export_chrome_trace(str(trace_file))
        print(f"\nProfiler trace saved to: {trace_file}")
        print("You can visualize it at: chrome://tracing")

        # 통계 출력
        print("\n" + "-" * 80)
        print("Top 10 operations by CPU time:")
        print("-" * 80)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        if self.device == "cuda":
            print("\n" + "-" * 80)
            print("Top 10 operations by CUDA time:")
            print("-" * 80)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print("\n" + "-" * 80)
        print("Top 10 operations by memory:")
        print("-" * 80)
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

        # 통계를 파일로 저장
        stats_file = output_path / "profiler_stats.txt"
        with open(stats_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Top operations by CPU time:\n")
            f.write("=" * 80 + "\n")
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            f.write("\n\n")

            if self.device == "cuda":
                f.write("=" * 80 + "\n")
                f.write("Top operations by CUDA time:\n")
                f.write("=" * 80 + "\n")
                f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                f.write("\n\n")

            f.write("=" * 80 + "\n")
            f.write("Top operations by memory:\n")
            f.write("=" * 80 + "\n")
            f.write(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))

        print(f"\nDetailed stats saved to: {stats_file}")

        return {
            "result": result[0]["generation"],
            "trace_file": str(trace_file),
            "stats_file": str(stats_file),
        }

    def benchmark_different_lengths(
        self,
        prompt_configs: List[Dict],
        max_gen_len: int = 100,
    ):
        """다양한 프롬프트 길이에 대한 벤치마크"""

        print("\n" + "=" * 80)
        print("Benchmarking different prompt lengths...")
        print("=" * 80)

        results = []

        for config in prompt_configs:
            prompt = config["prompt"]
            name = config["name"]

            print(f"\n{'-' * 80}")
            print(f"Testing: {name}")
            print(f"Prompt length: {len(self.llama.tokenizer.encode(prompt, bos=True, eos=False))} tokens")
            print(f"{'-' * 80}")

            result = self.benchmark_simple(
                prompts=[prompt],
                max_gen_len=max_gen_len,
                temperature=0.0,
            )

            result["name"] = name
            result["prompt_length"] = len(self.llama.tokenizer.encode(prompt, bos=True, eos=False))
            results.append(result)

            print(f"Elapsed time: {result['elapsed_time']:.3f}s")
            print(f"Generated tokens: {result['total_tokens']}")
            print(f"Throughput: {result['tokens_per_sec']:.2f} tokens/sec")
            print(f"Generated text: {result['results'][0][:100]}...")

        # 요약 테이블 출력
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Test Name':<30} {'Prompt Len':<12} {'Gen Tokens':<12} {'Time (s)':<12} {'Tokens/sec':<12}")
        print("-" * 80)

        for r in results:
            print(f"{r['name']:<30} {r['prompt_length']:<12} {r['total_tokens']:<12} {r['elapsed_time']:<12.3f} {r['tokens_per_sec']:<12.2f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Speed benchmark for Llama inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (if not set, uses LLAMA_MODEL_PATH env var or downloads)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Huggingface model name to download if model-path not provided",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=100,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for profiler results",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run detailed profiling with torch profiler",
    )

    args = parser.parse_args()

    # 모델 경로 결정
    model_path = args.model_path

    if model_path is None:
        # 환경 변수 체크
        model_path = os.environ.get("LLAMA_MODEL_PATH")

    if model_path is None:
        # 모델 다운로드
        print("Model path not provided. Attempting to download from Huggingface...")
        try:
            model_path = download_model(
                model_name=args.model_name,
                cache_dir="./models"
            )
        except Exception as e:
            print(f"\nFailed to download model: {e}")
            print("\nAlternatively, you can:")
            print("1. Provide --model-path argument")
            print("2. Set LLAMA_MODEL_PATH environment variable")
            print("3. Ensure you have Huggingface authentication set up")
            return

    print(f"\nUsing model from: {model_path}")

    # dtype 변환
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # 벤치마크 초기화
    benchmark = SpeedBenchmark(
        model_path=model_path,
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
    )

    # 워밍업
    benchmark.warmup(num_runs=3)

    # 다양한 길이의 프롬프트 테스트
    prompt_configs = [
        {
            "name": "Short prompt",
            "prompt": "Hello, how are you?",
        },
        {
            "name": "Medium prompt",
            "prompt": "Write a short story about a robot learning to paint. Include details about the robot's emotions and struggles.",
        },
        {
            "name": "Long prompt",
            "prompt": """You are a helpful AI assistant. Please provide a detailed explanation of how neural networks work,
            including the concepts of forward propagation, backpropagation, activation functions, and optimization algorithms.
            Make sure to explain it in a way that a beginner can understand, with examples where appropriate.""",
        },
    ]

    # 다양한 길이 벤치마크
    results = benchmark.benchmark_different_lengths(
        prompt_configs=prompt_configs,
        max_gen_len=args.max_gen_len,
    )

    # 결과 저장
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    results_file = output_path / "benchmark_results.json"
    with open(results_file, "w") as f:
        # results를 JSON 직렬화 가능하게 변환
        serializable_results = []
        for r in results:
            serializable_results.append({
                "name": r["name"],
                "prompt_length": r["prompt_length"],
                "total_tokens": r["total_tokens"],
                "elapsed_time": r["elapsed_time"],
                "tokens_per_sec": r["tokens_per_sec"],
                "generated_text": r["results"][0],
            })
        json.dump(serializable_results, f, indent=2)

    print(f"\nBenchmark results saved to: {results_file}")

    # Profiler 실행 (옵션)
    if args.profile:
        print("\n" + "=" * 80)
        print("Running detailed profiling...")
        print("=" * 80)

        profile_result = benchmark.benchmark_with_profiler(
            prompt="Write a short story about artificial intelligence.",
            max_gen_len=args.max_gen_len,
            temperature=0.0,
            output_dir=args.output_dir,
        )

        print(f"\nGenerated text: {profile_result['result']}")

    print("\n" + "=" * 80)
    print("Benchmark completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
