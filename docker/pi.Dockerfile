ARG BASE_IMAGE=basis-env
FROM ${BASE_IMAGE} AS basis-robot-env

# Swap back to root to run system installation
USER root

# python3-dev shouldn't technically be required, but pipca9685 is causing a build failure
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    wget https://archive.raspberrypi.org/debian/raspberrypi.gpg.key -O - | sudo apt-key add - && \
    echo  "deb http://archive.raspberrypi.com/debian/ bookworm main" > /etc/apt/sources.list.d/raspi.list && \
    apt update && \
    apt install -y --no-install-recommends \
        libcamera0.3 \
        libcamera-dev \
        libcamera-ipa \
        libi2c-dev \
        libevdev-dev \
        python3-jsonschema \
        python3-jinja2 \
        python-is-python3 \
        python3-dev \
        pkg-config

# Probably not right - this adds the basis user into the correct groups for my pi - this might not work on others
RUN groupadd -g 102 input && usermod -a -G input basis
RUN groupadd -g 994 i2c && usermod -a -G i2c basis
RUN usermod -a -G dialout basis

USER basis
ENV BASIS_PLATFORM=PI
