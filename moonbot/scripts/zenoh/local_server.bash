#!/bin/bash
set -e -o pipefail
# echo 'alias spedo="speedometer -t wlP1p1s0f0 -r wlP1p1s0f0 || speedometer -t wlp1s0f0 -r wlp1s0f0 || speedometer -t wlp2s0f0 -r wlp2s0f0 || speedometer -t wlp3s0f0 -r wlp3s0f0"' >> ~/.bashrc

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
WS="${SCRIPT_DIR}/../../hero_workspace"

source /opt/ros/jazzy/setup.bash || source /opt/ros/humble/setup.bash
source ${SCRIPT_DIR}/source.bash

# LIMB_NAMESPACE="$(python3 -c "from moonbot_launcher.modules.id_mapping import id2namespace; print(id2namespace())")"

ros2 run rmw_zenoh_cpp rmw_zenohd
# disown
#
# cat <<EOF
#
# ######### IMPORTANT #########
#
# Zenoh local server running in the background!
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
