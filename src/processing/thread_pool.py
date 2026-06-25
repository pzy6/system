import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Optional, Dict, List

logger = logging.getLogger(__name__)

class PriorityThreadPool:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PriorityPool")
        self.running = True
        self.priority_tasks: List[dict] = []
        self.lock = threading.Lock()
        
    def submit(self, func: Callable, *args, priority: int = 0, callback: Optional[Callable] = None, **kwargs) -> Future:
        future = self.executor.submit(func, *args, **kwargs)
        
        if callback:
            def wrapper(fut: Future):
                try:
                    result = fut.result()
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error: {str(e)}")
            
            future.add_done_callback(wrapper)
        
        return future

    def submit_high_priority(self, func: Callable, *args, callback: Optional[Callable] = None, **kwargs) -> Future:
        return self.submit(func, *args, priority=10, callback=callback, **kwargs)

    def submit_normal_priority(self, func: Callable, *args, callback: Optional[Callable] = None, **kwargs) -> Future:
        return self.submit(func, *args, priority=5, callback=callback, **kwargs)

    def submit_low_priority(self, func: Callable, *args, callback: Optional[Callable] = None, **kwargs) -> Future:
        return self.submit(func, *args, priority=1, callback=callback, **kwargs)

    def shutdown(self, wait: bool = True):
        self.running = False
        self.executor.shutdown(wait=wait)
        logger.info("PriorityThreadPool shutdown complete")

    def get_stats(self) -> dict:
        return {
            'running': self.running,
            'max_workers': self.executor._max_workers,
            'active_workers': len([t for t in threading.enumerate() if 'PriorityPool' in t.name])
        }

class TaskQueue:
    def __init__(self, maxsize: int = 100):
        self.queue: List[dict] = []
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)

    def put(self, item: dict, priority: int = 0):
        with self.lock:
            if len(self.queue) >= self.maxsize:
                self.queue.pop(0)
            
            insert_pos = 0
            for i, existing in enumerate(self.queue):
                if existing.get('priority', 0) <= priority:
                    insert_pos = i + 1
            
            self.queue.insert(insert_pos, {'item': item, 'priority': priority})
            self.not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> dict:
        with self.not_empty:
            if not self.queue:
                if timeout is None:
                    self.not_empty.wait()
                elif timeout > 0:
                    self.not_empty.wait(timeout)
                else:
                    raise Exception("Queue is empty")
            
            if self.queue:
                return self.queue.pop(0)['item']
            raise Exception("Queue is empty")

    def qsize(self) -> int:
        with self.lock:
            return len(self.queue)

    def empty(self) -> bool:
        return self.qsize() == 0