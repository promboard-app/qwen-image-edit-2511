# Qwen-Image-Edit-2511 — 말로 이미지 고치기 (서버급)

PromBoard 커뮤니티 플러그인입니다. 캔버스에 **Qwen-Image-Edit-2511 편집** 노드 하나를 더합니다. 고칠 이미지(1~3장)와 고치는 말(프롬프트)을 이으면 고친 이미지 한 장(PNG)을 냅니다. 여러 장을 이으면 한 장면으로 섞습니다(예: 왼쪽 곰과 오른쪽 곰을 공원 광장에 마주 세우기).

> **미검증 · 서버급입니다.** PromBoard 팀은 이 플러그인을 만들기만 하고 실제로 돌려 보지 않았습니다. 시험한 PC(RTX 4070 SUPER 12GB)에는 모델이 들어가지 않기 때문입니다. 결과 그림이 없는 것도 그래서입니다. VRAM 48GB 이상인 그래픽카드(RTX 6000 Ada · A6000 · L40S · A100 · H100 등)에서 쓰세요.

## 무엇을 하나

- 모델: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)(2025-12-17, 트랜스포머 20.4B). Qwen-Image-Edit-2509 를 고친 판으로, 인물 · 제품의 모습 유지, 여러 장 섞기, 산업 디자인 · 기하 추론이 나아졌다고 공식 카드가 밝힙니다.
- 입력: 고칠 이미지(필수, 1~3장) · 고치는 말(필수, 프롬프트 노드에서 잇기)
- 출력: 고친 이미지 1장. 크기는 마지막 이미지의 가로세로 비로 약 100만 픽셀(1024×1024 안팎)입니다.
- 설정
  - 단계 수: 공식 예시 40
  - 프롬프트 따르기: 공식 예시 4(true_cfg_scale). 1 이면 네거티브 프롬프트를 쓰지 않고 두 배 빠릅니다
  - 네거티브 프롬프트: 비우면 공식 예시처럼 빈칸 한 글자를 씁니다
  - 시드: -1 이면 매번 다른 결과

## 받는 것과 용량

공식 저장소의 bf16 가중치를 그대로 받습니다(줄인 판을 쓰지 않습니다). 모두 약 **57.7GB** 입니다.

| 부품 | 크기 |
|---|---|
| 트랜스포머(QwenImageTransformer2DModel, 20.4B) | 40.9GB |
| 글 · 그림 인코더(Qwen2.5-VL-7B) | 16.6GB |
| VAE · 토크나이저 · 설정 | 0.3GB |

## 필요한 그래픽카드

- **VRAM 48GB 이상**(서버급)이 필요합니다. 트랜스포머 40.9GB 를 한 번에 올려 계산하기 때문입니다. 공식 VRAM 수치는 없고, 이 값은 가중치 크기로 정한 근거값입니다.
- 플러그인은 부품을 VRAM 에 맞게 올립니다. 다 들어가면 전부 올리고, 아니면 글 인코더 → 트랜스포머 → VAE 를 차례로 쓸 때만 올립니다. 글 인코더가 혼자도 안 들어가면 조각씩 올립니다. 트랜스포머가 혼자 안 들어가면 "VRAM 48GB 이상이 필요하다"는 오류를 냅니다(조각씩 올리면 한 장에 수십 분이 걸려 막았습니다).
- 시스템 메모리(RAM)는 64GB 이상을 권합니다(가중치 57.7GB 를 메모리에 올립니다).
- NVIDIA(CUDA) 그래픽카드용입니다. CPU 로는 돌지 않습니다. 맥은 잠금 파일만 있고 돌려 보지 않았습니다.

## 라이선스와 조건

- 모델: Apache-2.0(HF 카드 license 칸). 상업 사용 · 지역 제한 조건이 없습니다.
- 이 플러그인 코드: Apache-2.0([LICENSE](LICENSE)).

## 실측 결과

미검증입니다. 이 플러그인으로 받거나 돌려 보지 않았습니다.

## 설치

1. PromBoard 앱 → 플러그인 매니저 → **커뮤니티 플러그인**에서 설치합니다.
2. 코드가 든 플러그인이라 **제한 모드**를 끄고 플러그인을 켭니다.
3. 처음 실행할 때 실행 준비가 파이썬 패키지(diffusers 0.42.0 · transformers 5.20.2 · accelerate 등, torch 는 앱이 고정한 2.8.0)와 모델(약 57.7GB)을 받습니다.
4. 캔버스 빈 곳을 우클릭하면 메뉴 맨 아래 플러그인 묶음에 **Qwen-Image-Edit-2511 편집**이 있습니다.

## English

A PromBoard community plugin that adds one node, **Qwen-Image-Edit-2511 편집** (edit). Connect 1–3 images and an edit instruction (prompt); it outputs one edited PNG. With several images it composes them into one scene.

> **Unverified, server-class.** The PromBoard team built this plugin but did not run it: the test PC (RTX 4070 SUPER, 12 GB) cannot hold the model, which is also why there are no result pictures. Use a GPU with 48 GB+ VRAM (RTX 6000 Ada, A6000, L40S, A100, H100, …).

- Model: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (2025-12-17, 20.4B transformer), the official bf16 weights as-is — about **57.7 GB** in total (transformer 40.9 GB, Qwen2.5-VL-7B encoder 16.6 GB, VAE/tokenizer/configs 0.3 GB).
- Output size: the aspect ratio of the last input image at about one megapixel.
- Settings: steps (official example 40), prompt adherence (true_cfg_scale, official example 4; 1 disables the negative prompt and runs twice as fast), negative prompt (empty = a single space, as in the official example), seed (-1 = random).
- VRAM: **48 GB or more**. There is no official number; this follows from loading the 40.9 GB transformer at once. The plugin keeps everything on the GPU when it fits, otherwise moves the encoder, transformer and VAE onto the GPU only while each runs, and streams the encoder block by block if even that does not fit. If the transformer alone does not fit it stops with an error instead of streaming it (that would take tens of minutes per image). 64 GB+ system RAM is recommended. NVIDIA CUDA only; CPU is blocked; the macOS lock file exists but is untested.
- License: model Apache-2.0 (Hugging Face license field); plugin code Apache-2.0.
- Measured results: none (unverified).

## License

Plugin code: Apache-2.0. Model weights are downloaded from their original source at first run under their own license (Apache-2.0).
