#!/bin/bash
set -e -o pipefail

if [ -z "$OPERATOR" ]; then
    echo "Error: OPERATOR environment variable is not defined."
    echo "Please set it with: export OPERATOR=your_name"
    echo "Or run with OPERATOR=your_name bash operator.bash"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export RCUTILS_CONSOLE_OUTPUT_FORMAT="{message}"
export RCUTILS_COLORIZED_OUTPUT=1
export RUST_LOG=off

# Cleanup function to kill background processes
cleanup() {
    echo "Shutting down..."
    wait "$PID_JOY" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

ketsid ros2 run joy joy_node --ros-args -r __ns:="/${OPERATOR}" -p deadzone:=0.025 -p autorepeat_rate:=0.0  2>/dev/null&
PID_JOY=$!
setpgid "$PID_JOY" "$PID_JOY" 2>/dev/null || true

ros2 run hero_ros2 operator \
    --ros-args \
    -p joint_speed:=0.15 \
    -p wheel_speed:=0.6 \
    -r /joy:="/${OPERATOR}/joy" \
    # -p translation_speed:=70 \
    # -p rotation_speed:=0.1 \
