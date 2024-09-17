
ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ONNXRUNTIME_COMMIT=v1.18.2
BUILD_CONFIG=Release
CPU_ARCHITECTURE=$(uname -m)

cd /tmp/

set -e
# clone onnxruntime repository and build
git clone ${ONNXRUNTIME_REPO} onnxruntime
export CC=
export CXX=
cd onnxruntime
git checkout ${ONNXRUNTIME_COMMIT}
./build.sh \
    --parallel \
    --build_shared_lib \
    --allow_running_as_root \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/${CPU_ARCHITECTURE}-linux-gnu/ \
    --use_tensorrt \
    --tensorrt_home /usr/lib/${CPU_ARCHITECTURE}-linux-gnu/ \
    --config ${BUILD_CONFIG} \
    --skip_tests \
    --cmake_extra_defines 'onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++'

# package and copy to output
ONNXRUNTIME_VERSION=$(cat /tmp/onnxruntime/VERSION_NUMBER)
rm -rf /tmp/onnxruntime/build/onnxruntime-linux-${CPU_ARCHITECTURE}-gpu-${ONNXRUNTIME_VERSION}
BINARY_DIR=build \
    ARTIFACT_NAME=onnxruntime-linux-${CPU_ARCHITECTURE}-gpu-${ONNXRUNTIME_VERSION} \
    LIB_NAME=libonnxruntime.so \
    BUILD_CONFIG=Linux/${BUILD_CONFIG} \
    SOURCE_DIR=/tmp/onnxruntime \
    COMMIT_ID=$(git rev-parse HEAD) \
    tools/ci_build/github/linux/copy_strip_binary.sh

cd /tmp/onnxruntime/build/onnxruntime-linux-${CPU_ARCHITECTURE}-gpu-${ONNXRUNTIME_VERSION}/lib/
ln -s libonnxruntime.so libonnxruntime.so.${ONNXRUNTIME_VERSION}
#cp -r /tmp/onnxruntime/build/onnxruntime-linux-${CPU_ARCHITECTURE}-gpu-${ONNXRUNTIME_VERSION} 
# TODO: add tar script
cd /tmp/onnxruntime/build
tar -czvf  onnxruntime-linux-${CPU_ARCHITECTURE}-gpu-${ONNXRUNTIME_VERSION}.tgz onnxruntime-linux-${CPU_ARCHITECTURE}-gpu-${ONNXRUNTIME_VERSION}/