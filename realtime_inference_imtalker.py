# realtime_inference_imtalker.py
# 实时音频消费 -> IMTalker 推理 -> 输出视频帧段到 global_frame_map / global_audio_frame_map
# 参考 navtalk Services/MuseTalk/scripts/realtime_inference.py 的循环与队列消费方式
#
# 实时性条件：每段音频 chunk_duration_sec（默认 0.5s）对应一段视频。
# 播放端以实时速度消费（0.5s 内容用 0.5s 播完），故需满足：
#   单段推理耗时 T_inference < chunk_duration_sec
# 若 T_inference < 0.5s，生成速度高于消费速度，可实时或略有缓冲；
# 若 T_inference > 0.5s，会落后，段间出现等待（卡顿/保持上一帧）。运行时会打印 [OK]/[LAG] 便于判断。

import os
import sys
import time
import tempfile
import traceback
from argparse import Namespace
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# 确保项目根在 path 中
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import realtime_share_state as share_state
from realtime_share_state import (
    global_audio_map,
    global_audio_lock,
    global_frame_map,
    global_audio_frame_map,
)

# 延迟导入 app，避免与 Gradio 冲突
def _get_imtalker_agent():
    import app as app_module
    if not hasattr(app_module, 'ensure_checkpoints'):
        raise RuntimeError("app module has no ensure_checkpoints")
    app_module.ensure_checkpoints()
    from app import AppConfig, InferenceAgent
    opt = AppConfig()
    if not os.path.exists(opt.renderer_path) or not os.path.exists(opt.generator_path):
        raise FileNotFoundError("IMTalker checkpoints not found.")
    return InferenceAgent(opt)


def _load_ref_image(avatar_path: str) -> Image.Image:
    """从路径加载参考图并转为 RGB PIL."""
    if not os.path.exists(avatar_path):
        raise FileNotFoundError(f"Avatar image not found: {avatar_path}")
    img = Image.open(avatar_path).convert("RGB")
    return img


def _pcm_24k_to_wav_16k(pcm_24k_bytes: bytes, wav_path: str, sr_orig: int = 24000, sr_target: int = 16000) -> None:
    """将 24kHz int16 PCM 重采样为 16kHz 并写入 WAV."""
    waveform = np.frombuffer(pcm_24k_bytes, dtype=np.int16)
    waveform = waveform.astype(np.float32) / 32768.0
    try:
        import librosa
        wav_16k = librosa.resample(waveform, orig_sr=sr_orig, target_sr=sr_target)
    except Exception:
        # 简单线性插值
        ratio = sr_target / sr_orig
        n_new = int(len(waveform) * ratio)
        wav_16k = np.interp(
            np.linspace(0, len(waveform) - 1, n_new),
            np.arange(len(waveform)),
            waveform,
        ).astype(np.float32)
    wav_16k = (np.clip(wav_16k, -1.0, 1.0) * 32767.0).astype(np.int16)
    import soundfile as sf
    sf.write(wav_path, wav_16k, sr_target, subtype="PCM_16")


def _video_path_to_frames_bgr(video_path: str) -> List[np.ndarray]:
    """读取 MP4 所有帧，返回 BGR 列表."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def main(args: Namespace) -> None:
    """
    主循环：从 global_audio_map[args.session_id] 取音频，按块推理，结果写入 global_frame_map / global_audio_frame_map.
    args 需包含: session_id, avatar_path, fps, crop, seed, nfe, cfg_scale, chunk_duration_sec, flush_timeout_sec
    """
    session_id = getattr(args, "session_id", None)
    avatar_path = getattr(args, "avatar_path", None)
    if not session_id or not avatar_path:
        print("[realtime_inference_imtalker] missing session_id or avatar_path")
        return

    fps = getattr(args, "fps", 25.0)
    crop = getattr(args, "crop", True)
    seed = getattr(args, "seed", 42)
    nfe = getattr(args, "nfe", 10)
    cfg_scale = getattr(args, "cfg_scale", 3.0)
    chunk_duration_sec = getattr(args, "chunk_duration_sec", 0.5)
    flush_timeout_sec = getattr(args, "flush_timeout_sec", 0.2)
    sample_rate_orig = 24000  # OpenAI 输出 24kHz
    sample_rate_imtalker = 16000
    chunk_samples_24k = int(sample_rate_orig * chunk_duration_sec)

    print("[realtime_inference_imtalker] Loading IMTalker model and ref image...")
    try:
        agent = _get_imtalker_agent()
        ref_pil = _load_ref_image(avatar_path)
    except Exception as e:
        print(f"[realtime_inference_imtalker] Init failed: {e}")
        traceback.print_exc()
        return

    accumulated_24k = np.array([], dtype=np.float32)
    last_audio_time = time.time()

    print(f"[realtime_inference_imtalker] session_id={session_id}, chunk={chunk_duration_sec}s, fps={fps}")

    while True:
        if share_state.should_stop:
            return

        with global_audio_lock:
            current_data = list(global_audio_map.get(session_id, []))
            if current_data:
                global_audio_map[session_id] = []

        if not current_data:
            # Flush 残留
            if accumulated_24k.size > 0 and (time.time() - last_audio_time) >= flush_timeout_sec:
                pad_len = chunk_samples_24k - accumulated_24k.size
                if pad_len > 0:
                    chunk_24k = np.pad(accumulated_24k, (0, pad_len), mode="constant", constant_values=0.0)
                else:
                    chunk_24k = accumulated_24k[:chunk_samples_24k]
                    accumulated_24k = accumulated_24k[chunk_samples_24k:]
                _run_one_chunk(
                    agent, ref_pil, chunk_24k, sample_rate_orig,
                    fps, crop, seed, nfe, cfg_scale, session_id,
                    chunk_duration_sec=chunk_duration_sec,
                )
                accumulated_24k = np.array([], dtype=np.float32)

            time.sleep(0.001)
            continue

        for audio_bytes in current_data:
            if share_state.should_stop:
                return
            if share_state.in_break:
                continue
            # int16 PCM 24k -> float32
            w = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            accumulated_24k = np.concatenate((accumulated_24k, w)) if accumulated_24k.size else w
            last_audio_time = time.time()

            while accumulated_24k.size >= chunk_samples_24k and not share_state.in_break:
                chunk_24k = accumulated_24k[:chunk_samples_24k].copy()
                accumulated_24k = accumulated_24k[chunk_samples_24k:]
                _run_one_chunk(
                    agent, ref_pil, chunk_24k, sample_rate_orig,
                    fps, crop, seed, nfe, cfg_scale, session_id,
                    chunk_duration_sec=chunk_duration_sec,
                )

        time.sleep(0.001)


def _run_one_chunk(
    agent,
    ref_pil: Image.Image,
    chunk_24k: np.ndarray,
    sample_rate_orig: int,
    fps: float,
    crop: bool,
    seed: int,
    nfe: int,
    cfg_scale: float,
    session_id: str,
    chunk_duration_sec: float = 0.5,
    log_timing: bool = True,
) -> None:
    """将一块 24k 波形写成 16k WAV，跑 IMTalker，结果写入 global_frame_map / global_audio_frame_map."""
    tmp_wav = None
    tmp_mp4 = None
    t0 = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        pcm_24k = (chunk_24k * 32767.0).astype(np.int16).tobytes()
        _pcm_24k_to_wav_16k(pcm_24k, tmp_wav, sr_orig=sample_rate_orig, sr_target=16000)
        t1 = time.perf_counter()

        tmp_mp4 = agent.run_audio_inference(
            ref_pil, tmp_wav, crop=crop, seed=seed, nfe=nfe, cfg_scale=cfg_scale
        )
        t2 = time.perf_counter()
        if not tmp_mp4 or not os.path.exists(tmp_mp4):
            return
        frames_bgr = _video_path_to_frames_bgr(tmp_mp4)
        t3 = time.perf_counter()
        if not frames_bgr:
            return
        with global_audio_lock:
            global_frame_map.append(frames_bgr)
            global_audio_frame_map.append(chunk_24k.astype(np.float32))
        if log_timing:
            elapsed = time.perf_counter() - t0
            ok = "OK" if elapsed < chunk_duration_sec else "LAG"
            prep = t1 - t0
            infer = t2 - t1
            read_back = t3 - t2
            print(
                f"[realtime_inference_imtalker] segment inference {elapsed:.2f}s (chunk={chunk_duration_sec}s) [{ok}] "
                f"| prep_wav={prep:.2f}s run_audio_inference={infer:.2f}s read_frames={read_back:.2f}s"
            )
    except Exception as e:
        print(f"[realtime_inference_imtalker] chunk error: {e}")
        traceback.print_exc()
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
        if tmp_mp4 and os.path.exists(tmp_mp4):
            try:
                os.remove(tmp_mp4)
            except Exception:
                pass
