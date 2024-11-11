ARG BASE_IMAGE=basis-env
FROM ${BASE_IMAGE} AS basis-robot-env

# Swap back to root to run system installation
USER root

# Manually download and add cerficate, as adv doesn't work on 3.6 for some reason.
RUN curl -L https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | apt-key add -

# Install OTA packages
RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.6 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/t194 r32.6 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
        
# TODO: it may be better to move this to a lower layer, so that we don't have CUDA on top all the time
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked <<EOF
    set -e

    cd /tmp 
    apt-get update
    apt-get -y --no-install-recommends install \
    tensorrt=8.6.2.3-1+cuda12.2 \
    libcudnn8-dev

    # TODO: hardcoded ubuntu22
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb

    apt-get update 
    
    apt list | grep nppplus
    apt-get -y --no-install-recommends install \
        nppplus-cuda-$(nvcc --version | sed -n "s/.*V\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\).*/\1/p" )
    
    rm *.deb
EOF

ARG ONNX_VERSION=1.18.2
# Install onnx gpu that we had to compile ourselves
RUN --mount=type=bind,source=docker/onnxruntime,target=/tmp/onnxruntime \
ls /tmp/onnxruntime && \
    tar -zxvf /tmp/onnxruntime/onnxruntime-linux-aarch64-gpu-${ONNX_VERSION}.tgz -C /usr/ onnxruntime-linux-aarch64-gpu-${ONNX_VERSION}/include onnxruntime-linux-aarch64-gpu-${ONNX_VERSION}/lib  --strip-components=1

USER basis
