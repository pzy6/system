import os
import time
import json
from queue import Empty, Full
from typing import Dict, Any

def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


def put_latest(target_queue, item) -> int:
    """非阻塞写入队列；队列满时丢弃旧数据，保留最新数据。"""
    try:
        target_queue.put_nowait(item)
        return 0
    except Full:
        dropped = 0
        while True:
            try:
                target_queue.get_nowait()
                dropped += 1
            except Empty:
                break
        try:
            target_queue.put_nowait(item)
            return dropped
        except Full:
            return dropped

def timestamp_to_datetime(timestamp: float) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

def save_json(data: Dict, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_current_timestamp() -> float:
    return time.time()

def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"