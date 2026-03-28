#!/bin/bash
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
. $SCRIPT_DIR/source.bash

ros2 launch realman_interface realman_interface_launch.py
