# realtime_ws_store.py
# 按 session_id 存储 WebSocket 连接，并支持向指定 session 发送消息、分发信令
from typing import Optional, Callable, Set
import asyncio


class WebSocketStore:
    clients = {}       # dict: session_id -> WebSocketResponse
    listeners = {}     # dict: session_id -> set(async_func)

    @classmethod
    def add(cls, key: str, session) -> None:
        cls.clients[key] = session

    @classmethod
    def remove(cls, key: str) -> None:
        cls.clients.pop(key, None)
        cls.listeners.pop(key, None)

    @classmethod
    def get(cls, key: str):
        return cls.clients.get(key)

    @classmethod
    async def send(cls, key: str, msg: str) -> None:
        ws = cls.clients.get(key)
        if ws and not ws.closed:
            await ws.send_str(msg)
        else:
            print(f"[WebSocketStore] session {key} 不存在或已断开")

    @classmethod
    def register_listener(cls, key: str, func: Callable) -> None:
        cls.listeners.setdefault(key, set()).add(func)

    @classmethod
    def unregister_listener(cls, key: str, func: Optional[Callable] = None) -> None:
        if key not in cls.listeners:
            return
        if func is None:
            cls.listeners.pop(key, None)
        else:
            cls.listeners[key].discard(func)
            if not cls.listeners[key]:
                cls.listeners.pop(key, None)

    @classmethod
    async def dispatch_message(cls, key: str, msg: str) -> None:
        funcs = cls.listeners.get(key)
        if not funcs:
            return
        for f in list(funcs):
            try:
                await f(key, msg)
            except Exception as e:
                print(f"[WebSocketStore] dispatch_message listener error: {e}")
