#!/bin/bash
set -e -o pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
WS="${SCRIPT_DIR}/../hero_workspace"

. ${WS}/install/setup.bash
export RCUTILS_CONSOLE_OUTPUT_FORMAT="{message}"
export RCUTILS_COLORIZED_OUTPUT=1

cd ${SCRIPT_DIR}/../
doit build

python3 -m moonbot_launcher.assemblies.output > ${SCRIPT_DIR}/output.urdf

echo "
URDF saved as ${SCRIPT_DIR}/output.urdf"
