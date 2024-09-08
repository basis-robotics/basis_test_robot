FROM basis-env AS basis-robot-env

# TODO: this depends on python, which is a bit sad
# we can probably untar this instead


USER root

# TODO: hardcoded ubuntu22
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked <<EOF
    set -e
    cd /tmp 
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update 
    apt-get -y install nppplus-cuda-$(nvcc --version | sed -n "s/.*V\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\).*/\1/p" )
    rm *.deb

    # TODO: we put tensorrt in the base container but it's going to be removed, swap back the base container and just install cudnn
    apt install -y --no-install-recommends tensorrt libcudnn8-dev
EOF

# Install the headers and CPU variants
# ARG ONNX_VERSION=1.19.2
# RUN <<EOF
#     cd /tmp
#     wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz
#     tar -zxvf onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz -C /usr/ onnxruntime-linux-aarch64-${ONNX_VERSION}/include onnxruntime-linux-aarch64-${ONNX_VERSION}/lib  --strip-components=1
#     rm *.tgz
# EOF
ARG ONNX_VERSION=1.18.2
RUN --mount=type=bind,source=docker/onnxruntime,target=/tmp/onnxruntime \
ls /tmp/onnxruntime && \
    tar -zxvf /tmp/onnxruntime/onnxruntime-linux-aarch64-gpu-${ONNX_VERSION}.tgz -C /usr/ onnxruntime-linux-aarch64-gpu-${ONNX_VERSION}/include onnxruntime-linux-aarch64-gpu-${ONNX_VERSION}/lib  --strip-components=1

# # Jetson only
# # https://elinux.org/Jetson_Zoo#ONNX_Runtime
# # Jetpack 6.0
# # Python 3.10
# RUN <<EOF
#     set -e
#     cd /tmp/
#     wget https://nvidia.box.com/shared/static/48dtuob7meiw6ebgfsfqakc9vse62sg4.whl -O onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl
#     pip3 install onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl
#     ln -s /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so /usr/lib/libonnxruntime_providers_cuda.so
#     ln -s /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/libonnxruntime_providers_tensorrt.so /usr/lib/libonnxruntime_providers_tensorrt.so
#     # This might already exist!
#     ln -s /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/libonnxruntime_providers_shared.so /usr/lib/libonnxruntime_providers_shared.so
#     # ln -s /usr/local/lib/python3.10/dist-packages/onnxruntime/capi/onnxruntime_pybind11_state.so /usr/lib/libonnxruntime_pybind11_state.so
#     rm *.whl
# EOF
# # Non Jetson GPU...
# https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/onnxruntime-cuda-12/PyPI/onnxruntime-gpu/overview/1.18.1
USER basis