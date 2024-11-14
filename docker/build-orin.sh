SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASIS_SOURCE_DIR=${BASIS_SOURCE_DIR:-"$SCRIPT_DIR/../../basis"}

BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.3.0

BASIS_ENABLE_ROS=${BASIS_ENABLE_ROS:-0}

BASIS_ENABLE_ROS=${BASIS_ENABLE_ROS} ${BASIS_SOURCE_DIR}/docker/build-env.sh  --build-arg BASE_IMAGE=${BASE_IMAGE}

ADDITIONAL_ARGS=""
if [ ${BASIS_ENABLE_ROS} -ne 0 ]; then
    ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --build-arg BASE_IMAGE=basis-env-ros"
fi

docker build --tag basis-robot-env --target basis-robot-env ${ADDITIONAL_ARGS} -f $SCRIPT_DIR/../docker/orin.Dockerfile $@ $SCRIPT_DIR/../
