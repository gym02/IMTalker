# realtime_imtalker_message_type.py
# WebSocket / WebRTC 消息类型（与前端约定一致，兼容 navtalk 前端）
from enum import Enum


class RealtimeMessageType(str, Enum):
    CONNECTED_SUCCESS = "conversation.connected.success"
    CONNECTED_FAIL = "conversation.connected.fail"
    CONNECTED_CLOSE = "conversation.connected.close"

    WEB_RTC_OFFER = "webrtc.signaling.offer"
    WEB_RTC_ANSWER = "webrtc.signaling.answer"
    WEB_RTC_ICE_CANDIDATE = "webrtc.signaling.iceCandidate"
    WEB_RTC_ICE_RESTART_REQUEST = "webrtc.signaling.iceRestartRequest"
    WEB_RTC_IN_BREAK = "webrtc.streaming.break"

    REALTIME_SESSION_CREATED = "realtime.session.created"
    REALTIME_SESSION_UPDATED = "realtime.session.updated"

    REALTIME_SPEECH_STARTED = "realtime.input_audio_buffer.speech_started"
    REALTIME_SPEECH_STOPPED = "realtime.input_audio_buffer.speech_stopped"
    REALTIME_CONVERSATION_ITEM_COMPLETED = (
        "realtime.conversation.item.input_audio_transcription.completed"
    )
    REALTIME_RESPONSE_AUDIO_TRANSCRIPT_DELTA = "realtime.response.audio_transcript.delta"
    REALTIME_RESPONSE_AUDIO_DELTA = "realtime.response.audio.delta"
    REALTIME_RESPONSE_AUDIO_TRANSCRIPT_DONE = "realtime.response.audio_transcript.done"
    REALTIME_RESPONSE_AUDIO_DONE = "realtime.response.audio.done"
    REALTIME_RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = (
        "realtime.response.function_call_arguments.done"
    )

    # 前端发送：用户音频追加
    REALTIME_INPUT_AUDIO_BUFFER_APPEND = "realtime.input_audio_buffer.append"
    REALTIME_INPUT_TEXT = "realtime.input.text"

    UNKNOWN_TYPE = "unknow"
