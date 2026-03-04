# IMTalker 实时流式模块 — 交接文档

## 1. 文档说明

- **用途**：实时数字人（OpenAI Realtime API + IMTalker + WebRTC）模块的交接与维护说明。
- **适用范围**：新接手的开发/运维需要了解实时链路、改配置、排查问题时使用。
- **与 app.py 的关系**：**未改写 app.py**。实时能力由一套**独立入口与新增模块**实现，仅**复用** `app.py` 中的 `InferenceAgent.run_audio_inference`；原有 Gradio 批处理（`python app.py`）保持不变，可单独运行。

---

## 2. 功能概述

| 能力 | 说明 |
|------|------|
| 实时对话 | 前端通过 WebSocket 连接服务端，用户可**语音或文本**与 OpenAI Realtime API 对话。 |
| 数字人视频 | 服务端将 AI 返回的**流式音频**按段送入 IMTalker 推理，生成口型视频，经 **WebRTC** 推流到浏览器播放。 |
| 外网访问 | 支持配置 Cloudflare TURN，便于经隧道/代理（如 `https://devrealtime.navtalk.ai/`）访问时建立 WebRTC。 |

**入口**：实时服务入口为 **`app_realtime_imtalker.py`**，与 `app.py` 并列，互不替换。

---

## 3. 模块与文件清单

### 3.1 后端（Python）

| 文件 | 职责 | 依赖关系 |
|------|------|----------|
| **app_realtime_imtalker.py** | 实时服务入口：aiohttp WebSocket、连接 OpenAI Realtime、写音频到共享状态、转发信令、提供 Demo 页路由。 | 依赖下方所有 realtime_* 及 app（仅间接通过 inference 使用） |
| **realtime_share_state.py** | 共享状态：`global_audio_map`、`global_frame_map` / `global_audio_frame_map`、`segment_index`、`should_stop`、`in_break`。 | 被 app_realtime、realtime_inference、realtime_publish 共用 |
| **realtime_inference_imtalker.py** | 推理循环：从 `global_audio_map` 按 chunk 取音频 → 写临时 WAV → 调用 **app.InferenceAgent.run_audio_inference** → 将结果写入 `global_frame_map` / `global_audio_frame_map`。 | **唯一直接依赖 app**（AppConfig、InferenceAgent） |
| **realtime_publish_imtalker.py** | WebRTC 推流：PeerConnection、offer/answer/ICE、按 `segment_index` 从 frame/audio map 取段并推送到前端。 | 依赖 realtime_share_state、realtime_ws_store、realtime_imtalker_message_type |
| **realtime_ws_store.py** | WebSocket 按 session_id 存储、发送消息、将 WebRTC 信令分发给 publish 的 listener。 | 被 app_realtime、realtime_publish 使用 |
| **realtime_imtalker_message_type.py** | 消息类型常量（与前端约定）：CONNECTED_SUCCESS、WEB_RTC_*、REALTIME_INPUT_* 等。 | 被 app_realtime、realtime_publish、前端引用 |

### 3.2 前端

| 文件 | 职责 |
|------|------|
| **realtime_demo.html** | 单页 Demo：连接 WebSocket、发送 CONNECTED_SUCCESS、处理 offer/answer/ICE、接收 WebRTC 音视频并播放；支持 TURN、ICE 重协商、videoReady（等 canplay 再显示视频）。 |

### 3.3 原有模块（仅被复用，未改逻辑）

| 文件 | 被使用方式 |
|------|------------|
| **app.py** | `realtime_inference_imtalker` 中：`from app import AppConfig, InferenceAgent`，循环内调用 `agent.run_audio_inference(ref_pil, tmp_wav, ...)`。 |

---

## 4. 架构与数据流

### 4.1 总体关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  浏览器 realtime_demo.html                                                    │
│  WebSocket + WebRTC (音视频)                                                  │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  app_realtime_imtalker.py (RealtimeIMTalkerWebSocket)                       │
│  · 建连后启动：推理线程 + 推流协程（区别见 4.3）                              │
│  · 收 CONNECTED_SUCCESS → 连 OpenAI，回 iceServers                          │
│  · 收 文本/音频 → 转发 OpenAI                                                │
│  · 收 OpenAI response.audio.delta → 写入 global_audio_map[session_id]        │
│  · WebRTC 信令 → WebSocketStore.dispatch_message → publish 模块              │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
┌───────────────────┐           ┌─────────────────────┐           ┌─────────────────────┐
│ realtime_share_   │           │ realtime_inference_ │           │ realtime_publish_   │
│ state             │           │ imtalker (线程)      │           │ imtalker (协程)      │
│                   │           │                     │           │                     │
│ global_audio_map  │◄── 写 ────│ app_realtime        │           │ 读 segment_index    │
│ global_frame_map  │◄── 写 ────│ 推理循环            │           │ 推 WebRTC           │
│ global_audio_     │◄── 写 ────│ run_audio_inference │           │                     │
│ frame_map         │           │ (app.InferenceAgent)│           │ 信令 ← dispatch     │
│ segment_index     │───────────│                     │───────────│                     │
│ should_stop       │           └─────────────────────┘           └─────────────────────┘
└───────────────────┘
```

### 4.2 数据流简述

1. **音频写入**：前端/OpenAI 的音频 → `app_realtime` 解码后写入 **global_audio_map[session_id]**。
2. **推理**：**realtime_inference** 从 `global_audio_map` 按 chunk（如 2s）取数据 → 写临时 WAV → **app.InferenceAgent.run_audio_inference** → 读 MP4 得到帧列表 → 写入 **global_frame_map** / **global_audio_frame_map**。
3. **推流**：**realtime_publish** 按 **segment_index** 从上述两 map 取段 → 通过 WebRTC 推送到前端 `<video>` / 音频播放。

### 4.3 为何推理用线程、推流用协程

**概念简答：**  
- **线程**：操作系统调度的“多段同时跑”的执行流，像多个工人一起干活；适合把会长时间占 CPU 的同步代码（如模型推理）放到单独线程，不卡住主程序。  
- **协程**：同一线程内用 `async/await` 写的“可暂停、可恢复”的任务，遇到 `await` 就让出执行权，由事件循环去跑别的协程；适合大量 I/O（网络、WebSocket），与信令等共用同一事件循环。

|  | 推理（realtime_inference） | 推流（realtime_publish） |
|--|----------------------------|---------------------------|
| **运行方式** | `asyncio.to_thread(realtime_inference_main, ...)` → **线程池**中执行 | `asyncio.create_task(publish_main_async(...))` → **事件循环**内协程 |
| **原因** | 推理是**同步阻塞**的：`while True` + `time.sleep()` + `agent.run_audio_inference()`（CPU/GPU 密集）。若放在主事件循环里会阻塞 WebSocket、信令、OpenAI 收发。 | **必须**与信令共用同一事件循环：前端发来的 offer/answer/ICE 由主循环的 WebSocket 收到后 `dispatch_message` 到 `streamer.on_message`，其中 `handle_answer`、`handle_ice_candidate` 等要操作 **RTCPeerConnection (pc)**；aiortc 的 pc 和 track 只能在一个 loop 里使用，推流的 `video_task` 操作的正是同一 pc 上的 track，故**不能**放到线程，必须用协程跑在主循环。 |
| **小结** | 线程：把“会阻塞”的同步逻辑放到线程里，不占满主循环。 | 协程：信令与 pc/track 绑定在主循环，推流必须与它们同属一个 loop，故采用协程是**必须**的，不是可选。 |

---

## 5. 环境与配置

### 5.1 依赖

- Python 3.x，项目原有依赖（torch、gradio、librosa、cv2、av、aiortc 等）。
- 环境与 **app.py** 一致；实时模块额外依赖：**aiohttp**、**dotenv**（以及 aiortc 用于 WebRTC）。
- 需安装 **ffmpeg**（推理与音视频处理）。
- 前端仅静态 HTML/JS，无需构建。

### 5.2 环境变量（.env）

在项目根目录配置 `.env`（可复制 `.env.example`，**勿将含密钥的 .env 提交版本库**）：

| 变量 | 说明 | 示例/默认 |
|------|------|-----------|
| OPENAI_WS_URL | OpenAI Realtime WebSocket 地址 | wss://api.openai.com/v1/realtime |
| OPENAI_REALTIME_MODEL | Realtime 模型名 | gpt-4o-realtime-preview-2024-12-17 |
| OPENAI_API_KEY | OpenAI API Key（前端也可通过 CONNECTED_SUCCESS 的 provider 传入） | sk-... |
| REALTIME_WS_HOST | 本机监听地址 | 0.0.0.0 |
| REALTIME_WS_PORT | 本机监听端口 | 8765 |
| REALTIME_TURN_KEY_ID | Cloudflare TURN Key ID（外网/隧道场景） | 可选 |
| REALTIME_TURN_API_TOKEN | Cloudflare TURN API Token | 可选 |
| REALTIME_TURN_TTL | TURN 凭证有效期（秒） | 86400 |

未配置 TURN 时仅使用 STUN；经外网访问时若 WebRTC 连不上，可配置 Cloudflare TURN 并确保前端使用返回的 iceServers。

---

## 6. 启动与运行

### 6.1 实时服务（推荐）

```bash
# 默认 host/port 可从 .env 读取
python app_realtime_imtalker.py

# 或显式指定
python app_realtime_imtalker.py --host 0.0.0.0 --port 8765
```

- 浏览器访问：`http://<host>:<port>/` 或 `http://<host>:<port>/realtime_demo.html`。
- WebSocket 连接地址：`ws://<host>:<port>/ws?session_id=<任意唯一串>&avatar_path=assets/source_1.png`（前端会按当前域名自动填）。

### 6.2 原有 Gradio 批处理（独立）

```bash
python app.py
```

与实时服务互不依赖，可单独运行。

---

## 7. 常见问题与注意事项

| 现象 | 可能原因 | 建议 |
|------|----------|------|
| 连接后无视频/黑屏 | 1）ICE 未 connected 就绑定了视频轨；2）后端先 failed 停推流 | 前端已做：ICE 为 connected 时再绑定并 play；videoReady 等 canplay/loadeddata 再显示。若仍无画面，看控制台是否出现「视频可播放，已显示」及后端是否报 ICE state=failed。 |
| ICE disconnected / failed | 网络或 TURN 不稳定；或后端在重协商时误停 | 已做：ICE 重协商期间不因 closed 调 stop；前端约 4s 后发 ICE 重协商请求。外网访问建议配 TURN。 |
| 首帧或长时间无画面 | 推理首段较慢或 TURN 建连慢 | 后端无 segment 时会推默认黑帧保持连接；可考虑用 avatar 图做首帧（当前未实现）。 |


### 7.1 日志与调试

- 推理耗时：`realtime_inference_imtalker` 会打印每段推理时间及 [OK]/[LAG]（是否超过 chunk 时长）。
- WebRTC：后端 publish 打印 Offer/Answer/ICE state；前端控制台有 ICE 状态及（若启用）候选对类型（relay/srflx/host）。
- aioice 日志较多时，已在 app_realtime_imtalker 中将 `aioice.ice` 设为 WARNING。

---

## 8. 后续可优化点（可选）

- **首帧优化**：用 avatar 图作为 WebRTC 首帧，减少黑屏时间（需在 publish 或 handle_connected_msg 中支持传入/加载头像图）。
- **推理性能**：若单段推理时间持续大于 chunk 时长，可考虑缩短 chunk、降低分辨率或优化模型/硬件。
- **TURN 与 ICE**：若频繁 disconnected，可增加候选对与连接状态的上报/监控，便于区分是 TURN 还是网络问题。

---

## 9. 附录：与 app.py 的复用关系（一句话）

**app.py 未被改成实时入口**；实时流式 = **新入口（app_realtime_imtalker）+ 共享状态（realtime_share_state）+ 推理循环（realtime_inference，按段调用 app.InferenceAgent.run_audio_inference）+ WebRTC 推流（realtime_publish）**，app.py 仅以“提供 InferenceAgent 与 run_audio_inference”的方式被复用。
