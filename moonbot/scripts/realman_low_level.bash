#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
. $SCRIPT_DIR/source.bash

ros2 launch realman_interface full_launch.py
