/**
 * 姿态后处理: COCO 17关键点 → PoseNet 18关键点映射 (C++ 优化)
 * 替换 _coco_to_posenet_keypoints() — 消除 72 dict 分配/人/帧
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// COCO index → PoseNet order (18 entries, -1 = synthesized)
static const int COCO_TO_POSENET[18] = {
    0,   // Nose
    -1,  // Neck (synthesized from shoulders)
    6,   // RShoulder
    8,   // RElbow
    10,  // RWrist
    5,   // LShoulder
    7,   // LElbow
    9,   // LWrist
    12,  // RHip
    14,  // RKnee
    16,  // RAnkle
    11,  // LHip
    13,  // LKnee
    15,  // LAnkle
    2,   // REye
    1,   // LEye
    4,   // REar
    3,   // LEar
};

py::array_t<float> coco_to_posenet_batch(
    py::array_t<float, py::array::c_style | py::array::forcecast> keypoints)
{
    auto buf = keypoints.request();
    int n_persons = static_cast<int>(buf.shape[0]);  // (N, 17, 3)
    const auto* ptr = static_cast<float*>(buf.ptr);

    auto result = py::array_t<float>({n_persons, 18, 3});
    auto res_buf = result.request();
    auto* out = static_cast<float*>(res_buf.ptr);

    for (int p = 0; p < n_persons; ++p) {
        const float* kp = ptr + p * 17 * 3;
        float* op = out + p * 18 * 3;

        for (int i = 0; i < 18; ++i) {
            int ci = COCO_TO_POSENET[i];
            if (ci >= 0 && ci < 17) {
                op[i * 3 + 0] = kp[ci * 3 + 0];  // x
                op[i * 3 + 1] = kp[ci * 3 + 1];  // y
                op[i * 3 + 2] = kp[ci * 3 + 2];  // confidence
            } else if (ci == -1) {
                // Neck = midpoint of left_shoulder (5) + right_shoulder (6)
                float lsx = kp[5 * 3 + 0], lsy = kp[5 * 3 + 1], lsc = kp[5 * 3 + 2];
                float rsx = kp[6 * 3 + 0], rsy = kp[6 * 3 + 1], rsc = kp[6 * 3 + 2];
                if (lsc > 0.1f && rsc > 0.1f) {
                    op[1 * 3 + 0] = (lsx + rsx) * 0.5f;
                    op[1 * 3 + 1] = (lsy + rsy) * 0.5f;
                    op[1 * 3 + 2] = (lsc + rsc) * 0.5f;
                }
            }
        }
    }
    return result;
}

PYBIND11_MODULE(pose_postprocess, m) {
    m.doc() = "COCO→PoseNet keypoint mapping (C++ optimized)";
    m.def("coco_to_posenet_batch", &coco_to_posenet_batch,
          py::arg("keypoints"));
}
