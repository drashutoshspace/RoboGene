#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source $SCRIPT_DIR/source.bash

if [ "$FASTRTPS_DEFAULT_PROFILES_FILE" != "" ]; then 
    export FASTRTPS_DEFAULT_PROFILES_FILE="${SCRIPT_DIR}/dds_profiles/normal_client.xml"
fi

ros2 launch motion_stack rviz_simu.launch.py
