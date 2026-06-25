# cython: language_level=3, boundscheck=False, wraparound=False
"""COCO 17→PoseNet 18 关键点批量映射 (Cython 优化)"""

import numpy as np
cimport numpy as np

# COCO → PoseNet 映射表
cdef int[18] COCO_TO_POSENET = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

def coco_to_posenet_batch(np.ndarray[np.float32_t, ndim=3] keypoints):
    """输入 (N, 17, 3) → 输出 (N, 18, 3)"""
    cdef int n = keypoints.shape[0]
    cdef np.ndarray[np.float32_t, ndim=3] result = np.zeros((n, 18, 3), dtype=np.float32)
    cdef float[:,:,:] kp = keypoints
    cdef float[:,:,:] out = result
    cdef int p, i, ci
    cdef float lsx, lsy, lsc, rsx, rsy, rsc

    for p in range(n):
        for i in range(18):
            ci = COCO_TO_POSENET[i]
            if ci >= 0 and ci < 17:
                out[p, i, 0] = kp[p, ci, 0]
                out[p, i, 1] = kp[p, ci, 1]
                out[p, i, 2] = kp[p, ci, 2]
            elif ci == -1:  # Neck = midpoint of shoulders
                lsx, lsy, lsc = kp[p, 5, 0], kp[p, 5, 1], kp[p, 5, 2]
                rsx, rsy, rsc = kp[p, 6, 0], kp[p, 6, 1], kp[p, 6, 2]
                if lsc > 0.1 and rsc > 0.1:
                    out[p, 1, 0] = (lsx + rsx) * 0.5
                    out[p, 1, 1] = (lsy + rsy) * 0.5
                    out[p, 1, 2] = (lsc + rsc) * 0.5
    return result
