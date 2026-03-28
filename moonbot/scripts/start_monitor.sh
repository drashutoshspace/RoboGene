#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source $SCRIPT_DIR/source.bash

export RCUTILS_CONSOLE_OUTPUT_FORMAT="{message}"

if [ "$ZENOH_SESSION_CONFIG_URI" != "" ]; then 
    source ${SCRIPT_DIR}/zenoh/source_robot.bash
fi

if [ "$FASTRTPS_DEFAULT_PROFILES_FILE" != "" ]; then 
    export FASTRTPS_DEFAULT_PROFILES_FILE="${SCRIPT_DIR}/dds_profiles/normal_client.xml"
fi

ros2 run moonbot_hero_monitor_ros2 monitor_node_ros2 --ros-args -r __ns:=/$LIMB_NAMESPACE
