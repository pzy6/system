"""音频告警工具 — 烟雾蜂鸣 + 摔倒语音"""
import threading
import logging

logger = logging.getLogger(__name__)


def play_smoke_beep():
    """烟雾检测蜂鸣（后台线程，不阻塞主循环）"""
    def _beep():
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(2000, 200)
        except Exception:
            pass  # 非 Windows 静默忽略
    threading.Thread(target=_beep, daemon=True).start()


def play_fall_voice():
    """摔倒语音告警 '老人摔倒了'（后台线程）"""
    def _speak():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
            engine.say("有人摔倒了")
            engine.runAndWait()
        except ImportError:
            # pyttsx3 未安装 → 回退蜂鸣
            try:
                import winsound
                for _ in range(5):
                    winsound.Beep(800, 400)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"TTS 播放失败: {e}")
    threading.Thread(target=_speak, daemon=True).start()
