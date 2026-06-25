cmake_minimum_required(VERSION 3.14)
project(silver_guardian_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11 (installed via pip)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")
find_package(pybind11 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# Module 1: Pose post-process
pybind11_add_module(pose_postprocess pose_postprocess.cpp)

# Module 2: Overlay renderer
pybind11_add_module(overlay_renderer overlay_renderer.cpp)
target_link_libraries(overlay_renderer PRIVATE OpenCV::core OpenCV::imgproc)

# Module 3: Frame preprocessor
pybind11_add_module(frame_preprocessor frame_preprocessor.cpp)
target_link_libraries(frame_preprocessor PRIVATE OpenCV::core OpenCV::imgproc)
