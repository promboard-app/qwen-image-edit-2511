# Qwen-Image-Edit-2511 — 말로 이미지 고치기

PromBoard 커뮤니티 플러그인입니다. 캔버스에 **Qwen-Image-Edit-2511 편집** 노드 하나를 더합니다. 고칠 이미지(1~3장)와 고치는 말(프롬프트)을 이으면 고친 이미지 한 장(PNG)을 냅니다. 여러 장을 이으면 한 장면으로 섞습니다(예: 왼쪽 곰과 오른쪽 곰을 공원 광장에 마주 세우기).

> **검증 안 됨 · 고성능.** PromBoard 팀은 이 플러그인을 만들기만 하고 실제로 돌려 보지 않았습니다. 시험한 PC(RTX 4070 SUPER 12GB)에는 가장 가벼운 판도 들어가지 않기 때문입니다. 결과 그림이 없는 것도 그래서입니다. 그래픽카드 메모리 20GB 이상(GGUF Q4 판)에서 쓰세요.

## 무엇을 하나

- 모델: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)(2025-12-17, 트랜스포머 20.4B). Qwen-Image-Edit-2509 를 고친 판으로, 인물 · 제품의 모습 유지, 여러 장 섞기, 산업 디자인 · 기하 추론이 나아졌다고 공식 카드가 밝힙니다.
- 입력: 고칠 이미지(필수, 1~3장) · 고치는 말(필수, 프롬프트 노드에서 잇기)
- 출력: 고친 이미지 1장. 크기는 마지막 이미지의 가로세로 비로 약 100만 픽셀(1024×1024 안팎)입니다.
- 설정
  - 경량화: 아래 표
  - 단계 수: 공식 예시 40
  - 프롬프트 따르기: 공식 예시 4(true_cfg_scale). 1 이면 네거티브 프롬프트를 쓰지 않고 두 배 빠릅니다
  - 네거티브 프롬프트: 비우면 공식 예시처럼 빈칸 한 글자를 씁니다
  - 시드: -1 이면 매번 다른 결과

## 경량화

노드의 **경량화** 알약에서 트랜스포머 정밀도를 고릅니다. 고른 판의 파일만 받습니다. **자동**은 이 PC 그래픽카드 메모리로 고릅니다 — 48GB 이상 원본, 28GB 이상 GGUF Q8, 그 밖은 GGUF Q4.

| 경량화 알약 | 트랜스포머 | 처음 받는 양 | 그래픽카드 메모리 |
|---|---|---|---|
| 원본 (bf16) | 공식 bf16 40.9GB | 57.72GB | 48GB 이상 |
| GGUF Q8 | [unsloth](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) Q8_0 21.8GB | 38.62GB | 28GB 이상 |
| GGUF Q4 | unsloth Q4_K_M 13.2GB | 30.10GB | 20GB 이상 |

- 공통 파일(16.86GB): 글 · 그림 인코더(Qwen2.5-VL-7B, bf16 16.6GB) · VAE · 토크나이저 · 설정. 판을 바꾸면 그 판의 트랜스포머만 더 받습니다.
- GGUF 판은 텐서 이름이 diffusers 트랜스포머와 같아 diffusers 0.42.0 의 GGUF 읽기 기능이 그대로 읽습니다(설정은 공식 transformer/config.json). 원본보다 메모리를 덜 쓰는 대신 결과가 조금 거칠 수 있습니다.
- 그래픽카드 메모리 값은 트랜스포머 크기에 계산 몫을 더한 어림값입니다(공식 수치 없음, 실측 아님).

## 필요한 그래픽카드 · 메모리

- 플러그인은 부품을 그래픽카드 메모리에 맞게 올립니다. 다 들어가면 전부 올리고, 아니면 글 인코더 → 트랜스포머 → VAE 를 쓸 차례에만 올립니다. 글 인코더가 혼자도 안 들어가면 조각씩 올립니다. 트랜스포머가 혼자 안 들어가면 오류를 냅니다(20B 를 조각씩 올리면 한 장에 수십 분이 걸려 막았습니다).
- 시스템 메모리(RAM)는 받는 양보다 넉넉하게(GGUF Q4 판 48GB 이상, 원본 판 64GB 이상)를 권합니다. 가중치를 모두 메모리에 올립니다.
- NVIDIA(CUDA) 그래픽카드용입니다. CPU 로는 돌지 않습니다. 리눅스 · 맥 잠금 파일도 있지만 돌려 보지 않았습니다.

## 라이선스와 조건

- 모델: Apache-2.0(공식 · unsloth GGUF 모두). 상업 사용 · 지역 제한 조건이 없습니다.
- 이 플러그인 코드: Apache-2.0([LICENSE](LICENSE)).

## 설치

1. PromBoard 앱(1.0.3 이상) → 플러그인 매니저 → **커뮤니티 플러그인**에서 설치합니다.
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. 처음 실행할 때 실행 준비가 파이썬 패키지(diffusers 0.42.0 · transformers 5.20.2 · accelerate · gguf, torch 는 앱이 고정한 2.8.0)와 고른 판의 모델을 받습니다.
4. 캔버스 빈 곳을 우클릭하면 메뉴 맨 아래 플러그인 묶음에 **Qwen-Image-Edit-2511 편집**이 있습니다.

## 바뀐 것

- 1.0.2: 부품을 올리는 코드(`nodes/gpu.py`)를 다른 이미지 플러그인과 같은 판으로 맞췄습니다(쓸 때만 올리는 부품 뒤에 조각씩 올리는 부품이 오면 앞 부품을 먼저 내림). 이 플러그인은 트랜스포머를 조각씩 올리지 않아 동작은 1.0.1 과 같습니다.
- 1.0.1: 경량화 설정(원본 bf16 · GGUF Q8 · GGUF Q4)과 라벨(기능 · 검증 안 됨)을 더했습니다. 필요한 그래픽카드 메모리가 48GB → 20GB(GGUF Q4)로 내려갔습니다. 앱 1.0.3 이상.
- 1.0.0: 처음 공개(원본 bf16 만).

## English

A PromBoard community plugin that adds one node, **Qwen-Image-Edit-2511 편집** (edit). Connect 1–3 images and an edit instruction (prompt); it outputs one edited PNG. With several images it composes them into one scene.

> **Unverified, high-end.** The PromBoard team built this plugin but did not run it: the test PC (RTX 4070 SUPER, 12 GB) cannot hold even the lightest option, which is also why there are no result pictures. Use a GPU with 20 GB+ VRAM (GGUF Q4).

- Model: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (2025-12-17, 20.4B transformer).
- **Quantization** setting picks the transformer precision; only the chosen files are downloaded. *Auto* picks bf16 on 48 GB+ VRAM, GGUF Q8 on 28 GB+, otherwise GGUF Q4.
  - bf16 (official, 40.9 GB; 57.72 GB download; ~48 GB VRAM) · [unsloth](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) GGUF Q8 (21.8 GB; 38.62 GB; ~28 GB) · GGUF Q4 (13.2 GB; 30.10 GB; ~20 GB). VRAM figures are estimates (no official numbers, not measured).
  - Shared files (16.86 GB): Qwen2.5-VL-7B encoder, VAE, tokenizer, configs. The GGUF tensors use diffusers' names, so diffusers 0.42.0 loads them with its GGUF loader.
- Output size: the aspect ratio of the last input image at about one megapixel.
- Settings: steps (official example 40), prompt adherence (true_cfg_scale, official example 4; 1 disables the negative prompt and runs twice as fast), negative prompt (empty = a single space, as in the official example), seed (-1 = random).
- Memory: everything goes on the GPU when it fits; otherwise each part is moved on only while it runs, and the encoder is streamed block by block if needed. If the transformer alone does not fit it stops with an error. 48 GB+ system RAM for Q4, 64 GB+ for bf16. NVIDIA CUDA only; CPU is blocked; Linux and macOS lock files exist but are untested.
- License: model Apache-2.0 (official and unsloth GGUF); plugin code Apache-2.0.
- Changes: 1.0.2 syncs `nodes/gpu.py` with the other image plugins (a part moved on per use is now moved off when a block-streamed part starts); this plugin never streams the transformer, so it behaves as 1.0.1; 1.0.1 adds the quantization setting and labels (app 1.0.3+), lowering the VRAM requirement from 48 GB to 20 GB; 1.0.0 was bf16 only.

## License

Plugin code: Apache-2.0. Model weights are downloaded from their original sources at first run under their own license (Apache-2.0).
