#!/bin/bash
set -e -o pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source $SCRIPT_DIR/source.bash

ROBOT_BRAND="$(python3 -c "from moonbot_launcher.modules.base import parse_id;
print(parse_id()[0]) if parse_id() is not None else print(None)")"

if [ "$ZENOH_SESSION_CONFIG_URI" != "" ]; then
    source ${SCRIPT_DIR}/zenoh/source_robot.bash
fi
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

if [ "$FASTRTPS_DEFAULT_PROFILES_FILE" != "" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${SCRIPT_DIR}/dds_profiles/normal_client.xml"
fi

if [ "$ROBOT_BRAND" == "None" ]; then
    echo "This is not a robot. Motors not started."
    exit 1
fi

if [ "$ROBOT_BRAND" == "hero" ]; then
    sudo ip link set can0 up type can bitrate 1000000 || true
    ros2 launch hero_epos auto.launch.py
else
    echo "Errors to ignore:
-    write: Input/output error
-    RTNETLINK answers: Device or resource busy"
    for dev in /dev/ttyACM*; do
        echo "scland appling on $dev"
        sudo slcand -o -c -s0 "$dev" || true
    done
    sleep 0.1
    sudo ip link set can0 up type can bitrate 250000 || true
    sleep 3 && bash $SCRIPT_DIR/gusta_motor_state/enable.bash &
    # for axis in 1 2 3 4; do
        # ros2 service call /${LIMB_NAMESPACE}/odrive_axis${axis}/request_axis_state odrive_can/srv/AxisState "{axis_requested_state: 8}" &
    # done
    ros2 launch gleg_commander auto.launch.py
fi

