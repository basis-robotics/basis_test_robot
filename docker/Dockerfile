FROM basis-env AS basis-robot-env

# TODO: this depends on python, which is a bit sad
# we can probably untar this instead
# https://github.com/microsoft/onnxruntime/releases/download/v1.6.0/onnxruntime-linux-x64-gpu-1.6.0.tgz
RUN pip install onnxruntime

USER root
# TODO: apt cache
RUN cd /tmp \ 
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get -y install nppplus-cuda-$(nvcc --version | sed -n "s/.*V\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\).*/\1/p" ) \
    && rm *.deb
USER basis