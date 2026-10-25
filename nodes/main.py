"""promboard.qwen-image-edit-2511 — Qwen-Image-Edit-2511(Qwen) 이미지 편집 노드 코드.

노드 하나(edit): 고칠 이미지(1~3장) + 고치는 말(프롬프트)을 받아 이미지 한 장(PNG)을 낸다.
  여러 장을 이으면 섞어서 한 장으로 만든다(예: 왼쪽 곰 + 오른쪽 곰을 한 장면에 — 공식 카드 예시).
  결과 크기는 마지막 이미지의 가로세로 비로 100만 픽셀(1024²) 안팎(파이프라인 기본값).
모델: 공식 Qwen/Qwen-Image-Edit-2511(Apache-2.0) — 트랜스포머 20.4B + 글 · 그림 인코더 Qwen2.5-VL-7B(bf16 16.6GB) + VAE.
경량화 설정(quantization) — 트랜스포머 정밀도를 고른다(manifest requires.models 의 when 이 고른 것만 받는다. '자동'은 서버가
  이 PC 그래픽카드 메모리로 풀어 넘긴다).
  bf16  공식 트랜스포머(40.9GB) — 그래픽카드 메모리 48GB 이상
  q8    unsloth/Qwen-Image-Edit-2511-GGUF Q8_0(21.8GB)
  q4    unsloth/Qwen-Image-Edit-2511-GGUF Q4_K_M(13.2GB)
  GGUF 판은 텐서 이름이 diffusers 트랜스포머와 같아 from_single_file 이 그대로 읽는다(설정은 공식 transformer/config.json).
공식 카드 예시 값: true_cfg_scale 4.0 · negative_prompt " " · 40단계 · guidance_scale 1.0.
🔴 이 플러그인은 PromBoard 팀이 만들기만 하고 실제로 돌려 보지 않았다(미검증 — 시험한 PC 가 12GB 라 모델이 들어가지 않는다).

실행: diffusers 0.42.0 QwenImageEditPlusPipeline + QwenImageTransformer2DModel(bf16 은 공식 저장소 그대로, GGUF 는
from_single_file + GGUFQuantizationConfig bf16 계산). 부품을 올리는 법은 gpu.py — 트랜스포머는 통째로 올릴 수 있을 때만
돌리고(20B 를 조각씩 올리기는 너무 느려 막았다), 글 인코더는 모자라면 조각씩 올린다.
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from . import gpu  # noqa: E402

REPO = "Qwen/Qwen-Image-Edit-2511"
GGUF = {
    # 경량화 설정 값: GGUF 모델 이름(<저장소>/<파일>). bf16 은 공식 저장소의 transformer/ 를 그대로 읽는다.
    "q8": "unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q8_0.gguf",
    "q4": "unsloth/Qwen-Image-Edit-2511-GGUF/qwen-image-edit-2511-Q4_K_M.gguf",
}
MAX_IMAGES = 3
# 부품이 계산할 때 더 쓰는 VRAM(1024² 기준). 공식 수치가 없어 Qwen-Image 계열 일반값으로 넉넉히 잡았다(미검증).
RESERVE_GIB = {"text_encoder": 1.0, "transformer": 3.0, "vae": 3.0}
SEED_MAX = 2 ** 31 - 1

_STATE = {"key": None, "pipe": None}


def load_pipeline(ctx, quant):
    import torch
    from diffusers import GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    repo_dir = os.path.dirname(os.path.dirname(ctx.model_path(f"{REPO}/scheduler/scheduler_config.json")))
    dtype = torch.bfloat16
    ctx.progress(0.03, "load-model")
    extra = {}
    if quant in GGUF:
        extra["transformer"] = QwenImageTransformer2DModel.from_single_file(
            ctx.model_path(GGUF[quant]),
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            dtype=dtype,
            config=repo_dir,
            subfolder="transformer",
            local_files_only=True,
        )
    ctx.progress(0.10, "load-model")
    pipe = QwenImageEditPlusPipeline.from_pretrained(repo_dir, dtype=dtype, local_files_only=True, **extra)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_pipeline(ctx, quant):
    key = (quant, ctx.device)
    if _STATE["key"] != key:
        drop_pipeline()
        pipe = load_pipeline(ctx, quant)
        ctx.progress(0.16, "load-model")
        reserve = {name: int(gib * gpu.GIB) for name, gib in RESERVE_GIB.items()}
        plan = gpu.place(pipe, ["text_encoder", "transformer", "vae"], ctx.device, reserve, no_group=("transformer",))
        ctx.log(f"placement {plan}")
        _STATE.update(key=key, pipe=pipe)
    return _STATE["pipe"]


def drop_pipeline():
    old = _STATE["pipe"]
    _STATE.update(key=None, pipe=None)
    if old is not None:
        gpu.release(old)


def load_images(ctx, paths):
    from PIL import Image, ImageOps

    out = []
    for p in paths:
        with Image.open(ctx.in_path(p)) as im:
            out.append(ImageOps.exif_transpose(im).convert("RGB"))
    return out


NEED = {"bf16": "48GB", "q8": "28GB", "q4": "20GB"}


def oom_message(quant):
    lighter = "" if quant == "q4" else " 경량화를 더 가벼운 판(GGUF Q4)으로 바꾸거나"
    return (
        f"그래픽카드 메모리(VRAM)가 모자라 편집하지 못했습니다. 고른 판은 VRAM {NEED[quant]} 이상에서 돕니다." + lighter
        + " 이미지 수를 줄이고, 그래픽카드를 쓰는 다른 프로그램을 닫은 뒤 다시 실행해 주세요."
    )


def edit(ctx, inputs, settings):
    import torch

    prompt = (inputs.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("고치는 말(프롬프트)을 이어 주세요(프롬프트 노드 → 프롬프트 칸).")
    paths = inputs.get("images") or []
    if not paths:
        raise ValueError("고칠 이미지를 하나 이상 이어 주세요.")
    if len(paths) > MAX_IMAGES:
        raise ValueError(f"이미지는 {MAX_IMAGES}장까지 이어 주세요(지금 {len(paths)}장).")
    if ctx.device == "cpu":
        raise RuntimeError("Qwen-Image-Edit-2511 은 큰 그래픽카드(가장 가벼운 GGUF Q4 판도 VRAM 20GB 이상)가 있어야 돕니다.")

    quant = settings["quantization"]
    steps = int(settings["steps"])
    cfg = float(settings["cfg"])
    negative = (settings.get("negative") or "").strip() or " "
    seed = int(settings["seed"])
    if seed < 0:
        seed = random.randint(0, SEED_MAX)
    images = load_images(ctx, paths)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        pipe = get_pipeline(ctx, quant)

        def on_step(_pipe, i, _t, kwargs):
            ctx.check_cancel()
            ctx.progress(0.2 + 0.75 * (i + 1) / steps, "denoise")
            return kwargs

        ctx.progress(0.18, "encode")
        image = pipe(
            image=images if len(images) > 1 else images[0],
            prompt=prompt,
            negative_prompt=negative,
            true_cfg_scale=cfg,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=on_step,
        ).images[0]
    except MemoryError as exc:
        ctx.log(f"placement failed: {exc}", "warn")
        drop_pipeline()
        raise RuntimeError(oom_message(quant)) from None
    except torch.OutOfMemoryError:
        drop_pipeline()
        raise RuntimeError(oom_message(quant)) from None
    except RuntimeError as exc:
        if not gpu.is_cpu_oom(exc):
            raise
        drop_pipeline()
        raise RuntimeError(gpu.RAM_MESSAGE) from None

    ctx.progress(0.98, "save")
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("parameters", f"{prompt}\nQwen-Image-Edit-2511 {quant} · steps {steps} · true_cfg {cfg} · seed {seed} · images {len(images)}")
    out = ctx.out_path(f"qwen-image-edit-{seed}.png")
    image.save(out, pnginfo=info)
    report = gpu.peak_report()
    if report:
        ctx.log(report)
    ctx.progress(1.0, "done")
    return {"image": out}


NODES = {"edit": edit}
