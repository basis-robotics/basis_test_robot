SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_ROOT=$SCRIPT_DIR/..

BASIS_SOURCE_DIR=${BASIS_SOURCE_DIR:-"$PROJECT_ROOT/../basis"}

if [ "$(docker ps -a -q -f name=basis)" ]; then
    docker exec -it basis /bin/bash $@
else
    # Note: this relies on macos specific user mapping magic to mount with the proper permissions
    docker run -w /basis_test_robot -v $PROJECT_ROOT:/basis_test_robot \
        -v $BASIS_SOURCE_DIR:/basis \
        -v $PROJECT_ROOT/../deterministic_replay:/deterministic_replay \
        --privileged --runtime=nvidia --net=host \
        --name basis --rm -it basis-robot-env /bin/bash $@
fi