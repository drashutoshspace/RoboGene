#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
WS="${SCRIPT_DIR}/../workspace"

LIMB_NAMESPACE="$(python3 -c "from moonbot_launcher.modules.id_mapping import id2namespace; print(id2namespace())")"

export RCUTILS_CONSOLE_OUTPUT_FORMAT="{message}"
export RCUTILS_COLORIZED_OUTPUT=1
