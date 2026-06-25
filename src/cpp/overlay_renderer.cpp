/**
 * 标注渲染: Native OpenCV C++ 绘制
 * 替换 _draw_overlays() — 消除 frame.copy() + Python O(N×M) 循环
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace py = pybind11;

void draw_overlays(
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> frame_arr,
    const std::vector<std::vector<int>>& face_bboxes,
    const std::vector<std::string>& face_labels,
    const std::vector<std::vector<int>>& fire_bboxes,
    const std::vector<std::vector<int>>& smoke_bboxes,
    py::array_t<float, py::array::c_style | py::array::forcecast> skeletons_arr,
    bool fall_detected, int fps)
{
    int h = static_cast<int>(frame_arr.shape(0));
    int w = static_cast<int>(frame_arr.shape(1));
    cv::Mat frame(h, w, CV_8UC3, const_cast<unsigned char*>(frame_arr.data()));

    // 人脸框 (绿色)
    for (size_t i = 0; i < face_bboxes.size(); ++i) {
        const auto& b = face_bboxes[i];
        cv::Rect r(b[0], b[1], b[2], b[3]);
        cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
        if (i < face_labels.size() && !face_labels[i].empty()) {
            cv::putText(frame, face_labels[i], cv::Point(b[0], b[1] - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }
    }

    // 火焰框 (橙色)
    for (const auto& b : fire_bboxes) {
        cv::Rect r(b[0], b[1], b[2], b[3]);
        cv::rectangle(frame, r, cv::Scalar(0, 165, 255), 2);
        cv::putText(frame, "FIRE", cv::Point(b[0], b[1] - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 165, 255), 2);
    }

    // 烟雾框 (红色)
    for (const auto& b : smoke_bboxes) {
        cv::Rect r(b[0], b[1], b[2], b[3]);
        cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "SMOKE", cv::Point(b[0], b[1] - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 255), 2);
    }

    // 骨架 — (N, 18, 3)
    auto skel_buf = skeletons_arr.request();
    int n_persons = static_cast<int>(skel_buf.shape[0]);
    const auto* skel_ptr = static_cast<float*>(skel_buf.ptr);

    // 13 对骨架连接线
    static const int PAIRS[13][2] = {
        {1,2}, {1,5}, {2,3}, {3,4}, {5,6}, {6,7},
        {1,8}, {8,9}, {9,10}, {1,11}, {11,12}, {12,13},
        {1,0}
    };

    for (int p = 0; p < n_persons; ++p) {
        const float* kp = skel_ptr + p * 18 * 3;
        for (int i = 0; i < 13; ++i) {
            int a = PAIRS[i][0], b = PAIRS[i][1];
            if (kp[a * 3 + 2] > 0.1f && kp[b * 3 + 2] > 0.1f) {
                cv::line(frame,
                         cv::Point(static_cast<int>(kp[a * 3]),
                                   static_cast<int>(kp[a * 3 + 1])),
                         cv::Point(static_cast<int>(kp[b * 3]),
                                   static_cast<int>(kp[b * 3 + 1])),
                         cv::Scalar(0, 255, 255), 2);
            }
        }
        for (int i = 0; i < 18; ++i) {
            if (kp[i * 3 + 2] > 0.1f) {
                cv::circle(frame,
                           cv::Point(static_cast<int>(kp[i * 3]),
                                     static_cast<int>(kp[i * 3 + 1])),
                           3, cv::Scalar(0, 255, 255), -1);
            }
        }
    }

    // 摔倒告警
    if (fall_detected) {
        cv::putText(frame, "FALL!", cv::Point(w / 2 - 60, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2,
                    cv::Scalar(0, 0, 255), 3);
    }

    // FPS 显示
    cv::putText(frame, "FPS:" + std::to_string(fps), cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
}

PYBIND11_MODULE(overlay_renderer, m) {
    m.doc() = "Native OpenCV overlay rendering (C++)";
    m.def("draw_overlays", &draw_overlays,
          py::arg("frame"), py::arg("face_bboxes"),
          py::arg("face_labels"), py::arg("fire_bboxes"),
          py::arg("smoke_bboxes"), py::arg("skeletons"),
          py::arg("fall_detected") = false, py::arg("fps") = 0);
}
