# MY OWN LLAMA

- 한장의 gpu에서 돌아갈 모델을 가정함 (no parallelism)
- 7B 모델 기준

## llama 구현하기

- [x] rotary embedding 
- [x] GQA
- [x] rms-normalization 
- [x] SwiGLU
- [x] feedforward
- [x] attention
- [x] transformer block
- [x] tokenizer
- [x] inference and generation

## How to run

```bash
pip install -r requirements.txt
python test_inference.py
```


## Naive 구현의 한계


### 1. 메모리 관리의 비효율성

  - Page Attention 같은 효율적 메모리 관리 필요

### 2. Attention 연산의 비효율성

#### 구현된 것
- Vanilla Self-Attention with GQA
- 기본적인 KV Cache

#### 구현할 것
- **Flash Attention**: O(N²) 메모리 사용 → O(N) 불가능
  - 현재: QK^T를 모두 메모리에 저장
  - 최적화 시: tiling과 recomputation으로 메모리 절약
  - 속도 차이: 2~4배 느림 (긴 시퀀스에서)

- **Fused Kernels**: 각 연산이 독립적으로 실행
  - Softmax + Scaling이 분리
  - RoPE가 별도 연산
  - 각 단계마다 GPU ↔ memory I/O 발생

- **Quantization**: FP16/FP32만 지원
  - INT8/INT4 quantization 없음
  - 메모리 사용량: 2~4배 더 많음

### 3. Generation의 비효율성

#### Autoregressive Generation
```python
# 현재 구현: 순차적 생성
for cur_pos in range(max_prompt_len, total_len):
    logits = self.model(tokens[:, cur_pos:cur_pos+1], ...)
    next_token = sample(logits)
```

**문제점:**
- 한 번에 하나의 토큰만 생성
- 각 스텝마다 전체 모델 forward pass
- Batch size=1로 제한하여 throughput 낮음

#### 없는 최적화 기법들

**Speculative Decoding**
- 작은 draft model로 여러 토큰 예측
- 큰 모델로 병렬 검증
- 2~3배 속도 향상 가능

**Continuous Batching**
- 여러 요청을 동적으로 배치 처리
- 현재: 요청마다 독립적 처리
- 최적화 시: GPU 활용률 극대화

**Beam Search / Sampling 최적화**
- 단순한 top-p sampling만 구현
- Beam search 없음
- Diverse beam search, constrained decoding 없음

### 4. 시스템 레벨 최적화 부재

#### Model Parallelism
```python
# 현재: 단일 GPU에 전체 모델 로드
model = model.to("cuda")
```

**한계:**
- Tensor Parallelism 없음: 큰 모델(70B+)은 단일 GPU에 불가능
- Pipeline Parallelism 없음: 레이어별 분산 처리 불가
- 다중 GPU 활용 불가

#### Kernel Optimization
- **CUDA Kernel**: 모든 연산이 PyTorch 기본 구현
  - Custom CUDA kernel 없음
  - Triton kernel 없음
  - 하드웨어 특성 활용 못함 (Tensor Core 등)

- **Operator Fusion**: 연산자 융합 없음
  ```python
  # 현재: 분리된 연산들
  x = self.norm(x)        # kernel 1
  x = self.attention(x)   # kernel 2
  x = x + residual        # kernel 3

  # 최적화 시: 하나의 fused kernel로
  ```

#### Memory I/O
- Weight loading: 순차적 로드 (병렬화 없음)
- Activation checkpointing: 미구현
- Mixed precision training: 추론만 FP16 지원

## TODO

### inference 관련 구현

- [ ] Page Attention (vLLM-style)
- [ ] Speculative Decoding
- [ ] Flash Attention 2,3
- [ ] Continuous Batching
- [ ] INT8 Quantization
- [ ] Kernel Fusion (Triton)
- [ ] 최적화이후 벤치마크