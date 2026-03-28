#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
. $SCRIPT_DIR/source.bash

ROBOT_BRAND="$(python3 -c "from moonbot_launcher.modules.base import parse_id;
print(parse_id()[0]) if parse_id() is not None else print(None)")"

if [ "$ROBOT_BRAND" != "hero" ]; then
    exit 1
fi

ROBOT_TYPE="$(python3 -c "from moonbot_launcher.modules.base import parse_id;
print(parse_id()[1]) if parse_id() is not None else print(None)")"

if [ "$ROBOT_TYPE" != "arm" ]; then
    exit 1
fi

ROBOT_VERSION="$(python3 -c "from moonbot_launcher.modules.base import parse_id;
print(parse_id()[2]) if parse_id() is not None else print(None)")"

if [ "$ROBOT_VERSION" == "1" ]; then
    ros2 run hero_ros2 calibration --ros-args --remap __ns:=/${LIMB_NAMESPACE}
fi

if [ "$ROBOT_VERSION" == "2" ]; then
    ros2 run hero_ros2 calibrationv2
fi

if [ "$ROBOT_VERSION" == "3" ]; then
    ros2 run hero_ros2 calibrationv2
fi

