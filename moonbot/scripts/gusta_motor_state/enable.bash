#!/bin/bash
set -e -o pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
source $SCRIPT_DIR/../source.bash

for axis in 1 2 3 4; do
    ros2 service call /${LIMB_NAMESPACE}/odrive_axis${axis}/request_axis_state odrive_can/srv/AxisState "{axis_requested_state: 8}"
done
