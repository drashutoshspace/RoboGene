#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
. $SCRIPT_DIR/source.bash

ros2 run hero_ros2 apriltag
