"""promboard.qwen-image-edit-2511 — Qwen-Image-Edit-2511(Qwen) 이미지 편집 노드 코드.

노드 하나(edit): 고칠 이미지(1~3장) + 고치는 말(프롬프트)을 받아 이미지 한 장(PNG)을 낸다.
  여러 장을 이으면 섞어서 한 장으로 만든다(예: 왼쪽 곰 + 오른쪽 곰을 한 장면에 — 공식 카드 예시).
  결과 크기는 마지막 이미지의 가로세로 비로 100만 픽셀(1024²) 안팎(파이프라인 기본값).
모델: 공식 Qwen/Qwen-Image-Edit-2511(Apache-2.0) bf16 그대로 — 트랜스포머 20.4B(40.9GB) + 글 · 그림 인코더
  Qwen2.5-VL-7B(16.6GB) + VAE, 모두 약 57.7GB. 줄인 판(GGUF)을 쓰지 않아 큰 그래픽카드(VRAM 48GB 이상 — 서버급)용이다.
공식 카드 예시 값: true_cfg_scale 4.0 · negative_prompt " " · 40단계 · guidance_scale 1.0.
🔴 이 플러그인은 PromBoard 팀이 만들기만 하고 실제로 돌려 보지 않았다(미검증 — 시험한 PC 가 12GB 라 모델이 들어가지 않는다).

실행: diffusers 0.42.0 QwenImageEditPlusPipeline. 부품을 올리는 법은 gpu.py — 트랜스포머(40.9GB)는 통째로 올릴 수 있을 때만
돌리고(조각씩 올리기는 너무 느려 막았다), 글 인코더는 모자라면 조각씩 올린다.
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from . import gpu  # noqa: E402

REPO = "Qwen/Qwen-Image-Edit-2511"
MAX_IMAGES = 3
# 부품이 계산할 때 더 쓰는 VRAM(1024² 기준). 공식 수치가 없어 Qwen-Image 계열 일반값으로 넉넉히 잡았다(미검증).
RESERVE_GIB = {"text_encoder": 1.0, "transformer": 3.0, "vae": 3.0}
SEED_MAX = 2 ** 31 - 1

_STATE = {"key": None, "pipe": None}


def load_pipeline(ctx):
    import torch
    from diffusers import QwenImageEditPlusPipeline

    repo_dir = os.path.dirname(os.path.dirname(ctx.model_path(f"{REPO}/scheduler/scheduler_config.json")))
    ctx.progress(0.03, "load-model")
    pipe = QwenImageEditPlusPipeline.from_pretrained(repo_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_pipeline(ctx):
    key = ctx.device
    if _STATE["key"] != key:
        drop_pipeline()
        pipe = load_pipeline(ctx)
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


def oom_message():
    return (
        "그래픽카드 메모리(VRAM)가 모자라 편집하지 못했습니다. Qwen-Image-Edit-2511 은 트랜스포머만 40.9GB 라 VRAM 48GB 이상"
        "(서버급 그래픽카드)이 있어야 돕니다."
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
        raise RuntimeError("Qwen-Image-Edit-2511 은 큰 그래픽카드(VRAM 48GB 이상)가 있어야 돕니다.")

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
        pipe = get_pipeline(ctx)

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
        raise RuntimeError(oom_message()) from None
    except torch.OutOfMemoryError:
        drop_pipeline()
        raise RuntimeError(oom_message()) from None

    ctx.progress(0.98, "save")
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("parameters", f"{prompt}\nQwen-Image-Edit-2511 · steps {steps} · true_cfg {cfg} · seed {seed} · images {len(images)}")
    out = ctx.out_path(f"qwen-image-edit-{seed}.png")
    image.save(out, pnginfo=info)
    report = gpu.peak_report()
    if report:
        ctx.log(report)
    ctx.progress(1.0, "done")
    return {"image": out}


NODES = {"edit": edit}
