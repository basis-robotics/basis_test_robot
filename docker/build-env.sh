SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASIS_SOURCE_DIR=${BASIS_SOURCE_DIR:-"$SCRIPT_DIR/../../basis"}

${BASIS_SOURCE_DIR}/docker/build-env.sh