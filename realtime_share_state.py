# realtime_share_state.py
# 实时推理与推流共享状态（与 navtalk MuseTalk 对齐）
from collections import deque
from threading import Lock
import time

# {session_id: [pcm_bytes, ...]}：实时音频片段缓冲（app_realtime 写入，realtime_inference 消费并清空）
global_audio_map = {}
global_audio_lock = Lock()
last_receive_time = {}  # {session_id: last_ts}

# 推理输出的视频帧段队列：每个元素是一个 segment 的 frames(list[np.ndarray] BGR)，供 WebRTC 推流
global_frame_map = deque()
# 推理输出的音频段队列：每个元素是一个 segment 的 waveform(np.ndarray, 24kHz float32)，与 global_frame_map 同步
global_audio_frame_map = deque()

segment_index = 0   # 推流消费的段索引
should_stop = False
in_break = False
