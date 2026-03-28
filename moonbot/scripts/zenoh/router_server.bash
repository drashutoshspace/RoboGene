#!/bin/bash
set -e -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
WS="${SCRIPT_DIR}/../../hero_workspace"

# source /opt/ros/jazzy/setup.bash || source /opt/ros/humble/setup.bash
source ${SCRIPT_DIR}/source.bash

export ZENOH_ROUTER_CONFIG_URI="${SCRIPT_DIR}/profile/router.json5"

# LIMB_NAMESPACE="$(python3 -c "from moonbot_launcher.modules.id_mapping import id2namespace; print(id2namespace())")"

ros2 run rmw_zenoh_cpp rmw_zenohd
# disown
#
# cat <<EOF
#
# ######### IMPORTANT #########
#
# Zenoh ROUTER server running in the background!
# Kill it using 'pkill zenoh'.
# Open logs with  'tail -f /tmp/zenoh.log'
# (Ctrl+C in here will not stop it)
#
# ######### IMPORTANT #########
#
# EOF
# #
# # sleep 1
# #
# # tail -f /tmp/zenoh.log
#
#
