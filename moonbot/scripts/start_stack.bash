#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
. $SCRIPT_DIR/source.bash

if [ -z "$LIMB_ID" ]; then  # LIMB_ID empty or not set
    echo "ERROR: \n LIMB_ID is not set in ~/.bashrc. Please run doit limb_id"
    exit 1
fi

if [ "$ZENOH_SESSION_CONFIG_URI" != "" ]; then 
    source ${SCRIPT_DIR}/zenoh/source_robot.bash
fi

if [ "$FASTRTPS_DEFAULT_PROFILES_FILE" != "" ]; then 
    export FASTRTPS_DEFAULT_PROFILES_FILE="${SCRIPT_DIR}/dds_profiles/normal_client.xml"
fi

ros2 launch moonbot_launcher robot.launch.py
