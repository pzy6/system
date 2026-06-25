/**
 * 帧预处理: OpenCV C++ 原生链路 (resize + CLAHE + GaussianBlur)
 * 替换 FrameProcessor.preprocess() 7 次 Python↔C++ FFI → 单次 C++ 调用
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

static cv::Ptr<cv::CLAHE> g_clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

py::array_t<unsigned char> preprocess(
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> frame_arr,
    int target_w, int target_h)
{
    int h = static_cast<int>(frame_arr.shape(0));
    int w = static_cast<int>(frame_arr.shape(1));
    cv::Mat frame(h, w, CV_8UC3, const_cast<unsigned char*>(frame_arr.data()));

    // 1. Resize
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(target_w, target_h));

    // 2. CLAHE on L channel
    cv::Mat lab;
    cv::cvtColor(resized, lab, cv::COLOR_BGR2LAB);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    g_clahe->apply(channels[0], channels[0]);
    cv::merge(channels, lab);
    cv::cvtColor(lab, resized, cv::COLOR_LAB2BGR);

    // 3. GaussianBlur
    cv::Mat output;
    cv::GaussianBlur(resized, output, cv::Size(3, 3), 0);

    // 4. Return as numpy array (must copy — output owns the data)
    auto result = py::array_t<unsigned char>({target_h, target_w, 3});
    auto res_buf = result.request();
    output.copyTo(cv::Mat(target_h, target_w, CV_8UC3, res_buf.ptr));
    return result;
}

PYBIND11_MODULE(frame_preprocessor, m) {
    m.doc() = "Frame preprocessing: resize+CLAHE+blur (C++)";
    m.def("preprocess", &preprocess,
          py::arg("frame"), py::arg("target_w") = 640, py::arg("target_h") = 480);
}
