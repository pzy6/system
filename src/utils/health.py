"""系统健康检查"""
import os, time, json, logging
from typing import Dict

logger = logging.getLogger(__name__)


def check_model_status(system) -> Dict:
    """检查所有模型加载状态"""
    return system.collect_model_status() if system else {}


def check_camera_status(system) -> Dict:
    """检查摄像头连接状态"""
    if not system or not system.camera_workers:
        return {"connected": 0, "total": 0, "details": []}
    details = []
    for w in system.camera_workers:
        details.append({
            "id": w.camera_id,
            "name": w.camera_name,
            "stats": w.get_stats() if hasattr(w, 'get_stats') else {},
        })
    return {
        "connected": sum(1 for d in details if d["stats"].get("connected", False)),
        "total": len(details),
        "details": details,
    }


def check_queue_health(system) -> Dict:
    """检查队列积压情况"""
    if not system:
        return {}
    queues = {
        "raw_frame": system.raw_frame_queue,
        "processed_frame": system.processed_frame_queue,
        "dashboard_frame": system.dashboard_frame_queue,
        "alarm": system.alarm_queue,
        "face_identity": system.face_identity_queue,
    }
    result = {}
    for name, q in queues.items():
        result[name] = {
            "size": q.qsize() if hasattr(q, 'qsize') else -1,
            "maxsize": q.maxsize if hasattr(q, 'maxsize') else -1,
        }
    return result


def check_disk_space(system) -> Dict:
    """检查存储目录磁盘空间"""
    import shutil
    paths = {}
    if system and system.config:
        storage = system.config.get("storage", {})
        for key in ["logs", "alarms", "screenshots", "videos"]:
            p = storage.get(key, "")
            if p:
                paths[key] = p

    result = {}
    for name, path in paths.items():
        try:
            usage = shutil.disk_usage(path)
            result[name] = {
                "free_gb": round(usage.free / 1024**3, 1),
                "total_gb": round(usage.total / 1024**3, 1),
                "used_pct": round((1 - usage.free / usage.total) * 100, 1),
            }
        except Exception:
            result[name] = {"error": "unavailable"}
    return result


def full_health_check(system) -> Dict:
    """完整健康检查"""
    return {
        "timestamp": time.time(),
        "models": check_model_status(system),
        "cameras": check_camera_status(system),
        "queues": check_queue_health(system),
        "disk": check_disk_space(system),
        "fps": system.fps if system else 0,
    }
