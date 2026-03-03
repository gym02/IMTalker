# app_realtime_imtalker.py
# 与 OpenAI Realtime API 连接交互，将 AI 流式音频实时送入 IMTalker 推理并 WebRTC 推流
# 参考 navtalk Services/MuseTalk/app_realtime.py 的 WebSocket、OpenAI 连接与消息转发
#
# 启动: python app_realtime_imtalker.py [--host 0.0.0.0] [--port 8765]
# 连接: ws://host:port/ws?session_id=xxx&avatar_path=assets/source_1.png
# 前端需先发 CONNECTED_SUCCESS，data.provider = { url, api_key, model }，再收 WEB_RTC_OFFER 做信令交换
# 用户音频/文本转发到 OpenAI，AI 的 response.audio.delta 写入队列，由 IMTalker 推理出视频帧，经 WebRTC 推流到前端

import os
import sys
import ssl
import json
import base64
import asyncio
import logging
import time
from argparse import Namespace
from typing import Optional

import aiohttp
import numpy as np
from aiohttp import web, WSMsgType
from dotenv import load_dotenv

load_dotenv()

# 项目根
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import realtime_share_state as share_state
from realtime_share_state import global_audio_map, last_receive_time, global_audio_lock
from realtime_imtalker_message_type import RealtimeMessageType
from realtime_ws_store import WebSocketStore
from realtime_inference_imtalker import main as realtime_inference_main
from realtime_publish_imtalker import main_async as publish_main_async

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 减少 aioice 的 ICE 候选检查刷屏，避免误以为卡在 "could not be resolved"
logging.getLogger("aioice.ice").setLevel(logging.WARNING)

# 从 .env 读取配置，缺省时使用下列默认值
OPENAI_WS_URL = os.getenv("OPENAI_WS_URL", "wss://api.openai.com/v1/realtime")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
REALTIME_WS_HOST = os.getenv("REALTIME_WS_HOST", "0.0.0.0")
REALTIME_WS_PORT = int(os.getenv("REALTIME_WS_PORT", "8765"))
# TURN（用于经 Cloudflare 等外网访问时 WebRTC 媒体中继），未设则仅用 STUN
REALTIME_TURN_URLS = os.getenv("REALTIME_TURN_URLS", "")  # 逗号分隔，如 turn:turn.cloudflare.com:3478,turns:turn.cloudflare.com:5349
REALTIME_TURN_USERNAME = os.getenv("REALTIME_TURN_USERNAME", "")
REALTIME_TURN_CREDENTIAL = os.getenv("REALTIME_TURN_CREDENTIAL", "")


def _get_ice_servers() -> list:
    """STUN + 可选的 TURN（从 .env 读取），供前端与 publish 端一致使用。"""
    ice = [{"urls": "stun:stun.l.google.com:19302"}]
    urls = (REALTIME_TURN_URLS or "").strip()
    username = (REALTIME_TURN_USERNAME or "").strip()
    credential = (REALTIME_TURN_CREDENTIAL or "").strip()
    if urls and username and credential:
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        if url_list:
            ice.append({"urls": url_list, "username": username, "credential": credential})
    return ice


def _get_ffmpeg_exe() -> str:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class RealtimeIMTalkerWebSocket:
    """
    实时语音 WebSocket 控制器：
    1. 管理客户端 WebSocket
    2. 连接 OpenAI Realtime API
    3. 将 response.audio.delta 写入 global_audio_map，供 IMTalker 推理消费
    4. 转发非音频消息到前端
    5. 处理 WebRTC 信令（由 publish 模块消费）
    """

    def __init__(self):
        self.session_id = ""
        self.avatar_path = ""
        self.client: Optional[aiohttp.ClientSession] = None
        self.client_session: Optional[web.WebSocketResponse] = None
        self.openai_websocket = None
        self.forward_openai_task: Optional[asyncio.Task] = None
        self.run_main_task: Optional[asyncio.Task] = None
        self.is_display_running = False

    async def connect_openai_wss(self, url: str, api_key: str, model: str) -> bool:
        if not url or not api_key:
            logger.error("OpenAI: URL or API Key empty")
            return False
        ws_url = f"{url.rstrip('/')}?model={model or OPENAI_REALTIME_MODEL}"
        logger.info("Connecting to OpenAI Realtime: %s", ws_url)
        api_key = api_key.strip().replace("\n", "").replace("\r", "").replace("\t", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:
            if not self.client or self.client.closed:
                self.client = aiohttp.ClientSession()
            self.openai_websocket = await self.client.ws_connect(
                ws_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            )
            logger.info("OpenAI Realtime API connected")
            self.is_display_running = True
            self.forward_openai_task = asyncio.create_task(self._forward_openai_to_client())
            return True
        except Exception as e:
            logger.error("OpenAI WebSocket failed: %s", e)
            self.openai_websocket = None
            return False

    async def _forward_openai_to_client(self):
        try:
            if not self.openai_websocket or self.openai_websocket.closed:
                return
            async for msg in self.openai_websocket:
                if share_state.should_stop or not self.is_display_running:
                    break
                if msg.type == WSMsgType.TEXT:
                    await self._process_openai_text(msg.data)
                elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("OpenAI forward error: %s", e)
        finally:
            logger.info("OpenAI forward task exited")

    async def _process_openai_text(self, message_data: str):
        try:
            data = json.loads(message_data)
            event_type = data.get("type")
            if event_type != "response.audio.delta":
                if self.client_session and not self.client_session.closed:
                    await self.client_session.send_str(message_data)
            if event_type == "response.audio.delta":
                await self._handle_audio_delta(data)
            elif event_type == "input_audio_buffer.speech_started":
                await self._break_in()
        except json.JSONDecodeError as e:
            logger.error("OpenAI message JSON error: %s", e)
        except Exception as e:
            logger.error("OpenAI message process error: %s", e)

    async def _handle_audio_delta(self, data: dict):
        try:
            delta = data.get("delta", "")
            if not delta:
                return
            audio_bytes = base64.b64decode(delta)
            sid = self.session_id
            with global_audio_lock:
                global_audio_map.setdefault(sid, []).append(audio_bytes)
            last_receive_time[sid] = time.time()
            share_state.in_break = False
        except Exception as e:
            logger.error("audio delta error: %s", e)

    async def _break_in(self):
        sid = self.session_id
        share_state.in_break = True
        with global_audio_lock:
            if sid in global_audio_map:
                global_audio_map[sid].clear()
        share_state.global_frame_map.clear()
        share_state.global_audio_frame_map.clear()
        share_state.segment_index = 0
        if sid in last_receive_time:
            del last_receive_time[sid]
        if self.client_session and not self.client_session.closed:
            await self.client_session.send_str(json.dumps({
                "type": RealtimeMessageType.WEB_RTC_IN_BREAK,
                "message": "",
            }))

    async def after_connection_established(self, request: web.Request, session: web.WebSocketResponse):
        query = request.query
        self.session_id = query.get("session_id") or query.get("sessionId")
        raw_avatar = query.get("avatar_path") or query.get("avatarPath") or query.get("name")
        if not self.session_id:
            await session.close(message="Missing session_id")
            return
        if not raw_avatar:
            await session.close(message="Missing avatar_path (ref image path or name)")
            return
        self.avatar_path = raw_avatar
        if not os.path.isabs(self.avatar_path):
            self.avatar_path = os.path.normpath(os.path.join(_PROJECT_ROOT, self.avatar_path))
        self.client_session = session
        WebSocketStore.add(self.session_id, session)
        logger.info("WebSocket connected session_id=%s avatar_path=%s", self.session_id, self.avatar_path)
        self.run_main_task = asyncio.create_task(
            self._run_background(self.session_id, self.avatar_path)
        )

    async def _run_background(self, session_id: str, avatar_path: str):
        args = Namespace(
            session_id=session_id,
            avatar_path=avatar_path,
            fps=25.0,
            crop=True,
            seed=42,
            nfe=10,
            cfg_scale=3.0,
            chunk_duration_sec=2.0,
            flush_timeout_sec=0.2,
        )
        logger.info("Starting IMTalker inference thread (model loading in background, may take 1-2 min on first run)...")
        asyncio.create_task(asyncio.to_thread(realtime_inference_main, args))
        asyncio.create_task(publish_main_async(avatar_path, session_id))
        if self.client_session and not self.client_session.closed:
            await self.client_session.send_str(json.dumps({
                "type": RealtimeMessageType.CONNECTED_SUCCESS,
                "message": "",
                "data": {
                    "session_id": session_id,
                    "iceServers": _get_ice_servers(),
                },
            }))
        share_state.should_stop = False

    async def after_connection_closed(self, session: web.WebSocketResponse):
        logger.info("Closing connection, cleaning up...")
        share_state.should_stop = True
        self.is_display_running = False
        self._clear_cache()
        if self.forward_openai_task and not self.forward_openai_task.done():
            self.forward_openai_task.cancel()
            try:
                await asyncio.wait_for(self.forward_openai_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if self.run_main_task and not self.run_main_task.done():
            self.run_main_task.cancel()
            try:
                await asyncio.wait_for(self.run_main_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        if self.openai_websocket and not self.openai_websocket.closed:
            try:
                await asyncio.wait_for(self.openai_websocket.close(), timeout=1.0)
            except Exception:
                pass
            self.openai_websocket = None
        if self.client_session:
            try:
                if not self.client_session.closed:
                    await self.client_session.close()
            except Exception:
                pass
            self.client_session = None
        if self.client and not self.client.closed:
            try:
                await self.client.close()
            except Exception:
                pass
            self.client = None
        WebSocketStore.remove(self.session_id)
        logger.info("Cleanup done")

    def _clear_cache(self):
        sid = self.session_id
        with global_audio_lock:
            if sid in global_audio_map:
                global_audio_map[sid].clear()
                del global_audio_map[sid]
            if sid in last_receive_time:
                del last_receive_time[sid]
        share_state.global_frame_map.clear()
        share_state.global_audio_frame_map.clear()
        share_state.segment_index = 0

    async def handle_message(self, session: web.WebSocketResponse, message: str):
        try:
            data = json.loads(message)
            event_type = data.get("type")
            if event_type == RealtimeMessageType.CONNECTED_SUCCESS:
                await self._handle_connection_success(data, message)
            elif event_type in (
                RealtimeMessageType.WEB_RTC_OFFER,
                RealtimeMessageType.WEB_RTC_ANSWER,
                RealtimeMessageType.WEB_RTC_ICE_CANDIDATE,
            ):
                await WebSocketStore.dispatch_message(self.session_id, message)
            elif event_type == RealtimeMessageType.REALTIME_INPUT_TEXT:
                await self._handle_input_text(data)
            elif event_type == RealtimeMessageType.REALTIME_INPUT_AUDIO_BUFFER_APPEND:
                await self._handle_input_audio_append(data)
            else:
                if self.openai_websocket and not self.openai_websocket.closed:
                    await self.openai_websocket.send_str(message)
                else:
                    logger.warning("No OpenAI connection, drop message type=%s", event_type)
        except json.JSONDecodeError as e:
            logger.error("handle_message JSON error: %s", e)
        except Exception as e:
            logger.error("handle_message error: %s", e)
            try:
                await session.close(message=str(e))
            except Exception:
                pass

    async def _handle_input_audio_append(self, data: dict):
        """将前端的 realtime.input_audio_buffer.append 转为 OpenAI 的 input_audio_buffer.append（API 要求参数名为 audio）。"""
        if not self.openai_websocket or self.openai_websocket.closed:
            return
        payload = data.get("data") or {}
        audio_b64 = payload.get("audio") or payload.get("delta") or ""
        if not audio_b64:
            return
        openai_event = {"type": "input_audio_buffer.append", "audio": audio_b64}
        try:
            await self.openai_websocket.send_str(json.dumps(openai_event))
        except Exception as e:
            logger.error("Send audio to OpenAI error: %s", e)

    async def _handle_input_text(self, data: dict):
        """将前端的 realtime.input.text 转为 OpenAI 的 conversation.item.create + response.create。"""
        if not self.openai_websocket or self.openai_websocket.closed:
            logger.warning("No OpenAI connection, drop text input")
            return
        payload = data.get("data") or {}
        text = (payload.get("content") or payload.get("text") or "").strip()
        if not text:
            return
        logger.info("Received text input, sending to OpenAI: %.60s", text)
        item_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        response_event = {"type": "response.create"}
        try:
            await self.openai_websocket.send_str(json.dumps(item_event))
            await self.openai_websocket.send_str(json.dumps(response_event))
        except Exception as e:
            logger.error("Send text to OpenAI error: %s", e)

    async def _handle_connection_success(self, data: dict, message: str):
        provider_data = data.get("data", {}).get("provider")
        if provider_data:
            url = provider_data.get("url") or OPENAI_WS_URL
            api_key = provider_data.get("api_key") or os.getenv("OPENAI_API_KEY", "")
            model = provider_data.get("model") or OPENAI_REALTIME_MODEL
            if api_key:
                await self.connect_openai_wss(url, api_key, model)
            else:
                logger.warning("No OpenAI API key in provider or OPENAI_API_KEY")
        await WebSocketStore.dispatch_message(self.session_id, message)


async def websocket_handler(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    session_id = request.query.get("session_id") or request.query.get("sessionId")
    logger.info("New WebSocket session_id=%s", session_id)
    rt = RealtimeIMTalkerWebSocket()
    try:
        await rt.after_connection_established(request, ws)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await rt.handle_message(ws, msg.data)
            elif msg.type == WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())
                break
    except asyncio.CancelledError:
        logger.info("WebSocket cancelled")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        try:
            await asyncio.wait_for(rt.after_connection_closed(ws), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("Cleanup timeout")
        except Exception as e:
            logger.error("Cleanup error: %s", e)
    return ws


async def serve_demo_page(_request: web.Request) -> web.Response:
    """提供 realtime_demo.html，便于通过 https://devrealtime.navtalk.ai/ 等访问。"""
    path = os.path.join(_PROJECT_ROOT, "realtime_demo.html")
    if not os.path.isfile(path):
        logger.warning("realtime_demo.html not found at %s", path)
        return web.Response(
            text=f"realtime_demo.html not found at {path}. Run from project root.",
            status=404,
            content_type="text/plain",
        )
    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    return web.Response(text=body, content_type="text/html")


def create_app():
    app = web.Application()
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/", serve_demo_page)
    app.router.add_get("/realtime_demo.html", serve_demo_page)
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="IMTalker Realtime WebSocket + OpenAI + WebRTC")
    parser.add_argument("--host", default=REALTIME_WS_HOST, help="Bind host (default from .env REALTIME_WS_HOST)")
    parser.add_argument("--port", type=int, default=REALTIME_WS_PORT, help="Bind port (default from .env REALTIME_WS_PORT)")
    args = parser.parse_args()
    app = create_app()
    logger.info("Realtime WebSocket server: ws://%s:%d/ws", args.host, args.port)
    logger.info("Demo page: http://%s:%d/ (or https://devrealtime.navtalk.ai/ via tunnel)", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
