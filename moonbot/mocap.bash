#!/bin/bash
set -e -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
WS="${SCRIPT_DIR}/hero_workspace"
doit build

. ${WS}/install/setup.bash
export RCUTILS_CONSOLE_OUTPUT_FORMAT="{message}"
export RCUTILS_COLORIZED_OUTPUT=1

ros2 run hero_ros2 mocap --ros-args -p simu:=True
