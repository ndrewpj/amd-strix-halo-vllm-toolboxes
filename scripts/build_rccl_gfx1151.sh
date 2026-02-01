#!/bin/bash
set -e
# Configuration
REPO_URL="https://github.com/kyuz0/rocm-systems.git"
BRANCH="gfx1151-rccl"
BUILD_DIR="build_gfx1151"
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
# Project sub-directory
PROJECT_DIR="projects/rccl"
echo "=== Building RCCL for gfx1151 ==="
echo "Repo: $REPO_URL"
echo "Branch: $BRANCH"
echo "ROCm Path: $ROCM_PATH"
# 1. Clone/Fetch
if [ -d "rocm-systems" ]; then
    echo "Directory 'rocm-systems' exists. Updating..."
    cd rocm-systems
    git fetch origin
    git checkout $BRANCH
    git pull origin $BRANCH
else
    echo "Cloning repository..."
    git clone -b $BRANCH $REPO_URL
    cd rocm-systems
fi
# 2. Setup Build Directory
echo "Entering project directory..."
cd $PROJECT_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR
echo "Configuring CMake for gfx1151..."
# We explicitly set GPU_TARGETS to gfx1151 to override the default list.
# We also set AMDGPU_TARGETS for standard rocm-cmake compliance.
CXX=$ROCM_PATH/bin/hipcc cmake .. \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/hipcc \
    -DDEFAULT_GPUS="gfx1151" \
    -DGPU_TARGETS="gfx1151" \
    -DAMDGPU_TARGETS="gfx1151" \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DBUILD_TESTS=OFF \
    -DGENERATE_SYM_KERNELS=OFF \
    -DENABLE_AMDSMI=OFF \
    -DCMAKE_BUILD_TYPE=Release
# 3. Build
echo "Building librccl.so..."
make -j$(nproc)
echo "=== Build Complete ==="
echo "Libraries are located in:"
echo "  $(pwd)/librccl.so"
echo "  $(pwd)/librccl.so.1"
