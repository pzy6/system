import cv2
import numpy as np
import threading
import time
import logging
from queue import Queue
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self, input_width: int = 640, input_height: int = 480):
        self.input_width = input_width
        self.input_height = input_height
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None
        self.processed_count = 0
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def set_input_queue(self, queue: Queue):
        self.input_queue = queue

    def set_output_queue(self, queue: Queue):
        self.output_queue = queue

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # YOLO 模型自带数据增强训练, 养老院固定摄像头无需 CLAHE
        return cv2.resize(frame, (self.input_width, self.input_height))

    def run(self):
        self.running = True
        logger.info("Frame processor started")
        
        while self.running:
            if self.input_queue is None or self.output_queue is None:
                time.sleep(0.1)
                continue
            
            try:
                item = self.input_queue.get(timeout=0.1)
                frame = item['frame']
                camera_id = item['camera_id']
                camera_name = item['camera_name']
                timestamp = item['timestamp']
                frame_count = item['frame_count']
                
                processed_frame = self.preprocess(frame)
                
                output_item = {
                    'camera_id': camera_id,
                    'camera_name': camera_name,
                    'frame': processed_frame,
                    'original_frame': frame,
                    'timestamp': timestamp,
                    'frame_count': frame_count
                }
                
                self.output_queue.put(output_item)
                self.processed_count += 1
                
                if self.processed_count % 1000 == 0:
                    logger.info(f"Processed {self.processed_count} frames")
                    
            except Exception as e:
                if not self.running:
                    break

    def start(self):
        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, name="FrameProcessor", daemon=True)
            self.thread.start()
            logger.info("Frame processor thread started")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        logger.info("Frame processor stopped")

    def get_stats(self) -> dict:
        return {
            'processed_count': self.processed_count,
            'input_width': self.input_width,
            'input_height': self.input_height,
            'running': self.running
        }