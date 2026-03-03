# realtime_publish_imtalker.py
# WebRTC 端到端视频推流：从 global_frame_map / global_audio_frame_map 取段，推送到前端
# 参考 navtalk Services/MuseTalk/publish.py 的轨道与推流逻辑

import asyncio
import traceback
import fractions
import json
import os
import time
from collections import deque
from typing import Optional, List, Callable

import cv2
import numpy as np
from aiortc import (
    RTCPeerConnection,
    RTCIceServer,
    RTCConfiguration,
    RTCSessionDescription,
    RTCIceCandidate,
    VideoStreamTrack,
    AudioStreamTrack,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame, AudioFrame

import realtime_share_state as share_state
from realtime_share_state import global_frame_map, global_audio_frame_map
from realtime_imtalker_message_type import RealtimeMessageType
from realtime_ws_store import WebSocketStore

# IMTalker 输出 512x512, 25fps, 音频 24kHz
DEFAULT_FPS = 25
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_VIDEO_SIZE = (512, 512)


def _normalize_audio_block(block: np.ndarray, chunk_samples: int) -> np.ndarray:
    """归一化音频块为 int16 PCM，长度不足则 pad."""
    block = np.asarray(block)
    if np.issubdtype(block.dtype, np.integer):
        if block.dtype != np.int16:
            block = block.astype(np.int16)
    else:
        block = block.astype(np.float32)
        block = np.clip(block, -1.0, 1.0)
        block = (block * 32767.0).astype(np.int16)
    if len(block) < chunk_samples:
        block = np.pad(block, (0, chunk_samples - len(block)), mode="constant")
    return block


class SingleFrameVideoStreamTrack(VideoStreamTrack):
    """单帧视频流轨道，用于 WebRTC 视频推流。"""

    def __init__(self, frame: Optional[np.ndarray] = None, fps: int = DEFAULT_FPS):
        super().__init__()
        h, w = DEFAULT_VIDEO_SIZE[1], DEFAULT_VIDEO_SIZE[0]
        self._current_frame = frame if frame is not None else np.zeros((h, w, 3), dtype=np.uint8)
        self._fps = fps

    async def recv(self):
        await asyncio.sleep(1 / self._fps)
        pts, time_base = await self.next_timestamp()
        if isinstance(self._current_frame, VideoFrame):
            frame = self._current_frame
        else:
            frame = VideoFrame.from_ndarray(self._current_frame, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    async def update_frame(self, new_frame):
        if isinstance(new_frame, VideoFrame):
            self._current_frame = new_frame.to_ndarray(format="bgr24")
        else:
            self._current_frame = np.asarray(new_frame)
            if self._current_frame.dtype != np.uint8:
                self._current_frame = np.clip(self._current_frame, 0, 255).astype(np.uint8)


class SingleFrameAudioStreamTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, channels: int = 1):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self._time_base = fractions.Fraction(1, sample_rate)
        self.audio_queue = deque(maxlen=100)
        self._samples_sent = 0

    async def recv(self):
        if self.readyState != "live":
            raise MediaStreamError
        while not self.audio_queue:
            await asyncio.sleep(0.001)
        pcm = self.audio_queue.popleft()
        samples = pcm.shape[0]
        frame = AudioFrame(format="s16", layout="mono", samples=samples)
        frame.sample_rate = self.sample_rate
        frame.time_base = self._time_base
        frame.planes[0].update(pcm.tobytes())
        frame.pts = self._samples_sent
        self._samples_sent += samples
        return frame

    def push_audio_data(self, pcm_int16: np.ndarray):
        self.audio_queue.append(pcm_int16)


class WebRTCStreamer:
    """WebRTC 流媒体控制器：管理 PeerConnection、信令、音视频推流。"""

    def __init__(
        self,
        send_ws_func: Callable,
        session_id: str,
    ):
        self.send_ws = send_ws_func
        self.target_session_id = session_id
        self.pc: Optional[RTCPeerConnection] = None
        self.video_track: Optional[SingleFrameVideoStreamTrack] = None
        self.audio_track: Optional[SingleFrameAudioStreamTrack] = None
        self.track_ready_event: Optional[asyncio.Event] = None
        self.media_task: Optional[asyncio.Task] = None
        self._closed = False
        self.start_time = time.time()
        # 默认黑帧 512x512；等待下一段时保持上一段最后一帧，减少断断续续感
        self.default_frame = np.zeros((DEFAULT_VIDEO_SIZE[1], DEFAULT_VIDEO_SIZE[0], 3), dtype=np.uint8)
        self.last_video_frame = None

    async def create_pc(self, ice_servers: Optional[List[dict]] = None):
        if self.pc and self.pc.connectionState not in ("closed", "failed"):
            await self.pc.close()
        ice_list = []
        if ice_servers:
            try:
                for s in ice_servers:
                    urls = s.get("urls")
                    if isinstance(urls, list):
                        for u in urls:
                            kw = {"urls": u}
                            if s.get("username") is not None:
                                kw["username"] = s["username"]
                            if s.get("credential") is not None:
                                kw["credential"] = s["credential"]
                            ice_list.append(RTCIceServer(**kw))
                    else:
                        ice_list.append(RTCIceServer(**s))
            except Exception as e:
                print(f"[publish] ICE config error: {e}")
        if not ice_list:
            ice_list = [RTCIceServer(urls="stun:stun.l.google.com:19302")]
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_list))
        self._closed = False

        @self.pc.on("connectionstatechange")
        async def _on_connectionstatechange():
            if self.pc is None:
                return
            if self.pc.connectionState in ("failed", "closed") and not self._closed:
                print(f"[publish] PeerConnection state={self.pc.connectionState}, stopping (often due to NAT when using Cloudflare Tunnel)")
                asyncio.create_task(self.stop())

        @self.pc.on("iceconnectionstatechange")
        async def _on_iceconnectionstatechange():
            if self.pc is None:
                return
            state = getattr(self.pc, "iceConnectionState", None)
            # 仅在 failed/closed 时关闭；disconnected 可能是暂时断线，不要立刻 stop 否则无法恢复
            if state in ("closed", "failed") and not self._closed:
                print(f"[publish] ICE state={state}, stopping (browser cannot reach server for media - need TURN if using https://devrealtime.navtalk.ai/)")
                asyncio.create_task(self.stop())

        @self.pc.on("icecandidate")
        async def _on_icecandidate(candidate):
            await self._send_ice_candidate(candidate)

        return self.pc

    async def _send_ice_candidate(self, candidate):
        try:
            if candidate is None:
                await self.send_ws(self.target_session_id, json.dumps({
                    "type": RealtimeMessageType.WEB_RTC_ICE_CANDIDATE,
                    "data": {"candidate": None},
                }))
            else:
                await self.send_ws(self.target_session_id, json.dumps({
                    "type": RealtimeMessageType.WEB_RTC_ICE_CANDIDATE,
                    "data": {
                        "candidate": {
                            "candidate": candidate.to_sdp(),
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex,
                        }
                    },
                }))
        except Exception as e:
            print(f"[publish] send ICE candidate error: {e}")

    def attach_tracks(
        self,
        video_track: SingleFrameVideoStreamTrack,
        audio_track: SingleFrameAudioStreamTrack,
    ):
        if not self.pc:
            raise RuntimeError("PeerConnection not created")
        self.video_track = video_track
        self.audio_track = audio_track
        self.pc.addTrack(video_track)
        self.pc.addTrack(audio_track)
        if self.track_ready_event:
            self.track_ready_event.set()

    async def create_and_send_offer(self):
        if not self.pc:
            raise RuntimeError("PeerConnection not created")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self.send_ws(self.target_session_id, json.dumps({
            "type": RealtimeMessageType.WEB_RTC_OFFER,
            "data": {
                "sdp": {
                    "type": self.pc.localDescription.type,
                    "sdp": self.pc.localDescription.sdp,
                }
            },
        }))
        print("[publish] Offer sent")

    async def handle_offer(self, data: dict):
        if not self.pc:
            return
        sdp = data.get("data", {}).get("sdp")
        if not sdp:
            return
        offer = RTCSessionDescription(sdp=sdp.get("sdp"), type=sdp.get("type"))
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await self.send_ws(self.target_session_id, json.dumps({
            "type": RealtimeMessageType.WEB_RTC_ANSWER,
            "data": {
                "sdp": {
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type,
                }
            },
        }))
        print("[publish] Answer sent")

    async def handle_answer(self, data: dict):
        if not self.pc:
            return
        sdp = data.get("data", {}).get("sdp")
        if not sdp:
            return
        answer = RTCSessionDescription(sdp=sdp.get("sdp"), type=sdp.get("type"))
        await self.pc.setRemoteDescription(answer)
        print("[publish] Remote answer applied")

    async def handle_ice_restart_request(self):
        """前端请求 ICE 重协商时，重新发 offer，缓解 disconnected 断断续续。"""
        if not self.pc or self._closed:
            return
        if self.pc.connectionState in ("closed", "failed"):
            return
        try:
            print("[publish] ICE restart: sending new offer")
            await self.create_and_send_offer()
        except Exception as e:
            print(f"[publish] ICE restart error: {e}")

    async def handle_ice_candidate(self, data: dict):
        if not self.pc:
            return
        cand_obj = data.get("data", {}).get("candidate")
        if cand_obj is None:
            return
        try:
            if isinstance(cand_obj, dict):
                sdp = cand_obj.get("candidate")
                parsed = candidate_from_sdp(sdp)
                ice = RTCIceCandidate(
                    component=parsed.component,
                    foundation=parsed.foundation,
                    ip=parsed.ip,
                    port=parsed.port,
                    priority=parsed.priority,
                    protocol=parsed.protocol,
                    type=parsed.type,
                    relatedAddress=parsed.relatedAddress,
                    relatedPort=parsed.relatedPort,
                    tcpType=parsed.tcpType,
                    sdpMid=cand_obj.get("sdpMid"),
                    sdpMLineIndex=cand_obj.get("sdpMLineIndex"),
                )
                await self.pc.addIceCandidate(ice)
        except Exception as e:
            print(f"[publish] add ICE candidate error: {e}")

    async def on_message(self, key: str, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        mtype = data.get("type")
        if mtype == RealtimeMessageType.CONNECTED_SUCCESS:
            await self.handle_connected_msg(data)
        elif mtype == RealtimeMessageType.WEB_RTC_OFFER:
            await self.handle_offer(data)
        elif mtype == RealtimeMessageType.WEB_RTC_ANSWER:
            await self.handle_answer(data)
        elif mtype == RealtimeMessageType.WEB_RTC_ICE_CANDIDATE:
            await self.handle_ice_candidate(data)
        elif mtype == RealtimeMessageType.WEB_RTC_ICE_RESTART_REQUEST:
            await self.handle_ice_restart_request()
        elif mtype == RealtimeMessageType.WEB_RTC_IN_BREAK:
            pass  # 可选：清空本地队列

    async def handle_connected_msg(self, data: dict):
        ice_servers = data.get("data", {}).get("iceServers") or data.get("iceServers")
        await self.create_pc(ice_servers=ice_servers)
        self.last_video_frame = None
        self.video_track = SingleFrameVideoStreamTrack(frame=self.default_frame, fps=DEFAULT_FPS)
        self.audio_track = SingleFrameAudioStreamTrack(sample_rate=DEFAULT_SAMPLE_RATE, channels=1)
        self.video_track._start_time = time.monotonic()
        self.video_track._time_base = fractions.Fraction(1, 90000)
        self.audio_track._time_base = fractions.Fraction(1, DEFAULT_SAMPLE_RATE)
        self.audio_track._samples_sent = 0
        self.audio_track.audio_queue.clear()
        self.attach_tracks(self.video_track, self.audio_track)
        await self.create_and_send_offer()

    async def push_av_segment(self, segment_index: int):
        if segment_index >= len(global_frame_map) or segment_index >= len(global_audio_frame_map):
            return
        frames = global_frame_map[segment_index]
        waveform = global_audio_frame_map[segment_index]
        sample_rate = DEFAULT_SAMPLE_RATE
        fps = DEFAULT_FPS
        audio_chunk_duration = 0.01
        chunk_samples = int(sample_rate * audio_chunk_duration)
        total_samples = len(waveform)
        total_duration = total_samples / sample_rate
        start_time = time.monotonic()

        task_a = asyncio.create_task(
            self._push_audio(waveform, start_time, chunk_samples, sample_rate, audio_chunk_duration)
        )
        task_v = asyncio.create_task(
            self._push_video(frames, start_time, total_duration, fps)
        )
        await task_a
        task_v.cancel()
        try:
            await task_v
        except asyncio.CancelledError:
            pass
        if frames:
            self.last_video_frame = np.asarray(frames[-1])
        print(f"[publish] Segment {segment_index} done, frames={len(frames)}")

    async def _push_audio(
        self,
        waveform: np.ndarray,
        start_time: float,
        chunk_samples: int,
        sample_rate: int,
        audio_chunk_duration: float,
    ):
        total_samples = len(waveform)
        pos = 0
        idx = 0
        while pos < total_samples and not share_state.in_break and not share_state.should_stop:
            target = start_time + idx * audio_chunk_duration
            now = time.monotonic()
            if target - now > 0:
                await asyncio.sleep(target - now)
            end = min(pos + chunk_samples, total_samples)
            block = waveform[pos:end]
            pcm = _normalize_audio_block(block, chunk_samples)
            if self.audio_track:
                self.audio_track.push_audio_data(pcm)
            pos = end
            idx += 1

    async def _push_video(
        self,
        frames: list,
        start_time: float,
        total_duration: float,
        fps: int,
    ):
        total_frames = len(frames)
        total_webrtc_frames = int(total_duration * fps)
        frame_idx = 0
        last_img = None
        webrtc_frame_count = 0
        while webrtc_frame_count < total_webrtc_frames and not share_state.in_break and not share_state.should_stop:
            target = start_time + (webrtc_frame_count / fps)
            now = time.monotonic()
            if target - now > 0:
                await asyncio.sleep(target - now)
            if frame_idx < total_frames:
                current_img = frames[frame_idx]
                frame_idx += 1
            else:
                current_img = last_img if last_img is not None else (frames[-1] if frames else self.default_frame)
            last_img = current_img
            if self.video_track:
                vf = VideoFrame.from_ndarray(current_img, format="bgr24")
                await self.video_track.update_frame(vf)
            webrtc_frame_count += 1

    async def video_task(self):
        await self.track_ready_event.wait()
        print("[publish] video_task started")
        try:
            while not share_state.should_stop:
                has_video = len(global_frame_map) > 0 and share_state.segment_index < len(global_frame_map)
                has_audio = len(global_audio_frame_map) > 0 and share_state.segment_index < len(global_audio_frame_map)
                if has_video and has_audio:
                    await self.push_av_segment(share_state.segment_index)
                    share_state.segment_index += 1
                else:
                    if self.video_track:
                        hold_frame = self.last_video_frame if self.last_video_frame is not None else self.default_frame
                        await self.video_track.update_frame(hold_frame)
                    await asyncio.sleep(1 / DEFAULT_FPS)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[publish] video_task error: {e}")
            traceback.print_exc()
        finally:
            print("[publish] video_task ended")

    def start_media_task(self):
        if self.media_task and not self.media_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        self.track_ready_event = asyncio.Event()
        self.media_task = loop.create_task(self.video_task())

    async def stop_media_task(self):
        if not self.media_task:
            return
        if not self.media_task.done():
            self.media_task.cancel()
            try:
                await self.media_task
            except asyncio.CancelledError:
                pass
        self.media_task = None

    async def _close_pc(self):
        pc = self.pc
        self.pc = None
        if pc and pc.connectionState not in ("closed", "failed"):
            try:
                await pc.close()
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[publish] close pc error: {e}")

    async def stop(self):
        share_state.should_stop = True
        await self.stop_media_task()
        await self._close_pc()
        self._closed = True
        print("[publish] Streamer stopped")


async def main_async(avatar_path: str, session_id: str):
    """异步主入口：创建 Streamer，注册信令监听，启动推流任务。"""
    streamer = WebRTCStreamer(
        send_ws_func=WebSocketStore.send,
        session_id=session_id,
    )
    WebSocketStore.register_listener(session_id, streamer.on_message)
    streamer.start_media_task()
    try:
        while not share_state.should_stop:
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[publish] main_async error: {e}")
        traceback.print_exc()
    finally:
        await streamer.stop()
        print(f"[publish] main_async done: {session_id}")


def main(args):
    """同步入口：asyncio.run(main_async(...))."""
    asyncio.run(main_async(args.avatar_path, args.session_id))
