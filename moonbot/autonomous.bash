#!/bin/bash
set -e -o pipefail

export RCUTILS_CONSOLE_OUTPUT_FORMAT="{message}"
export RCUTILS_COLORIZED_OUTPUT=1
export RUST_LOG=off

cleanup() {
    echo "Shutting down..."
    kill -KILL -- -"$PID_KEY" 2>/dev/null  || true
    wait "$PID_KEY" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

setsid ros2 run keyboard_event keyboard --ros-args -r __ns:="/auto"  2>/dev/null&
PID_KEY=$!
setpgid "$PID_KEY" "$PID_KEY" 2>/dev/null || true

ros2 run hero_ros2 self_assembly
