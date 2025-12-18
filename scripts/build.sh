# 1. Go to your build directory (or create a new one)
cd /home/jp4474/viaABC/src/viaABC/spatial2D/
# OR

mkdir -p build && cd build

# 2. Run CMake using g++
cmake .. \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

# 3. Compile
make -j$(nproc)
