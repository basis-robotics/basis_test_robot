SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASIS_SOURCE_DIR=${BASIS_SOURCE_DIR:-"$SCRIPT_DIR/../../basis"}

BASE_IMAGE=nvcr.io/nvidia/l4t-cuda:12.2.12-devel

BASIS_ENABLE_ROS=0 ${BASIS_SOURCE_DIR}/docker/build-env.sh  --build-arg BASE_IMAGE=${BASE_IMAGE}

docker build --tag basis-robot-env --target basis-robot-env -f $SCRIPT_DIR/../docker/Dockerfile $@ $SCRIPT_DIR/../