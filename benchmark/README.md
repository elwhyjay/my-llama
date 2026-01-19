# Speed Benchmark

torch profiler를 사용한 Llama inference 속도 벤치마킹 도구

## 기능

- **전체 생성 속도 측정**: tokens/sec
- **다양한 프롬프트 길이 테스트**: short, medium, long prompts
- **torch profiler 통합**: 레이어별 시간 분석, 병목 지점 파악
- **메모리 사용량 분석**
- **Chrome trace 생성**: chrome://tracing에서 시각화 가능

## 사용법

### 기본 벤치마크 실행 (환경변수 사용)

```bash
# LLAMA_MODEL_PATH 환경변수 설정
export LLAMA_MODEL_PATH=/path/to/your/model

python benchmark/speed_benchmark.py \
    --device cuda \
    --dtype float16 \
    --max-gen-len 100
```

### 직접 모델 경로 지정

```bash
python benchmark/speed_benchmark.py \
    --model-path /path/to/your/model \
    --device cuda \
    --dtype float16 \
    --max-gen-len 100
```

### Huggingface에서 자동 다운로드

```bash
# Huggingface 인증 필요: huggingface-cli login
python benchmark/speed_benchmark.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --device cuda \
    --dtype float16 \
    --max-gen-len 100
```

### Profiler와 함께 실행

```bash
python benchmark/speed_benchmark.py \
    --profile \
    --device cuda \
    --dtype float16 \
    --max-gen-len 100
```

### CPU에서 실행

```bash
python benchmark/speed_benchmark.py \
    --device cpu \
    --dtype float32 \
    --max-gen-len 50
```

## 옵션

- `--model-path`: 모델 경로 (기본값: None, LLAMA_MODEL_PATH 환경변수 사용 또는 자동 다운로드)
- `--model-name`: Huggingface 모델 이름 (기본값: meta-llama/Llama-2-7b-hf)
- `--device`: 디바이스 (cuda/cpu, 기본값: cuda)
- `--dtype`: 모델 dtype (float32/float16/bfloat16, 기본값: float16)
- `--max-seq-len`: 최대 시퀀스 길이 (기본값: 512)
- `--max-gen-len`: 최대 생성 길이 (기본값: 100)
- `--output-dir`: 결과 저장 디렉토리 (기본값: ./benchmark_results)
- `--profile`: torch profiler를 사용한 상세 분석 활성화

## 모델 경로 우선순위

1. `--model-path` 인자
2. `LLAMA_MODEL_PATH` 환경변수
3. Huggingface에서 자동 다운로드 (`--model-name` 사용)

## 출력 파일

벤치마크 실행 후 `benchmark_results/` 디렉토리에 다음 파일들이 생성됩니다:

- `benchmark_results.json`: 벤치마크 수치 결과
- `profiler_trace.json`: Chrome trace 파일 (--profile 옵션 사용 시)
- `profiler_stats.txt`: 상세 profiler 통계 (--profile 옵션 사용 시)

### Chrome Trace 시각화

1. Chrome 브라우저에서 `chrome://tracing` 열기
2. `profiler_trace.json` 파일 로드
3. 타임라인에서 각 연산의 시간과 의존성 확인

## 측정 항목

### 기본 벤치마크

- **Elapsed time**: 전체 소요 시간
- **Total tokens**: 생성된 토큰 수
- **Tokens/sec**: 초당 생성 토큰 수

### Profiler 벤치마크

- **CPU time**: 각 연산의 CPU 시간
- **CUDA time**: 각 연산의 GPU 시간
- **Memory usage**: 메모리 사용량
- **Operation breakdown**: 레이어별, 연산별 시간 분석

## 예제 출력

```
================================================================================
SUMMARY
================================================================================
Test Name                      Prompt Len   Gen Tokens   Time (s)     Tokens/sec
--------------------------------------------------------------------------------
Short prompt                   7            45           2.341        19.22
Medium prompt                  24           82           4.123        19.89
Long prompt                    68           95           5.234        18.15
```

## 병목 지점 분석

Profiler 결과에서 다음을 확인할 수 있습니다:

- **Attention 연산**: Q, K, V projection, scaled dot-product attention
- **FeedForward 연산**: SwiGLU, linear projections
- **Normalization**: RMSNorm
- **메모리 복사**: device-to-device, host-to-device transfers
- **Kernel launches**: CUDA kernel overhead

## 최적화 가이드

Profiler 결과를 분석하여 다음을 최적화할 수 있습니다:

1. **가장 많은 시간을 소비하는 연산 식별**
2. **메모리 복사 최소화**
3. **Fusion 기회 찾기** (여러 연산을 하나로 합치기)
4. **배치 처리 최적화**
5. **KV 캐시 효율성 개선**
