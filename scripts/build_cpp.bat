@echo off
echo 编译C++骨骼检测模块...

set PYTHONPATH=%PYTHONPATH%;%cd%\src

if not exist build mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH=%CONDA_PREFIX% -DPYTHON_EXECUTABLE=%CONDA_PREFIX%\python.exe ..

cmake --build . --config Release

cd ..

echo 编译完成!