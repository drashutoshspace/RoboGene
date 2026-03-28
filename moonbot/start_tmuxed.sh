#!/bin/bash
set -e -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
WS="${SCRIPT_DIR}/../workspace"

tmux kill-session -t moonbot || true
tmux new-session -s moonbot -d # -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID
# tmux set update-environment ROS_DOMAIN_ID
# tmux set-environment -g ROS_DOMAIN_ID $ROS_DOMAIN_ID
tmux set -g mouse on

# spec: 
# ┌───────────────┬───────────────────────────┐
# │ pane 0        │ pane 3                    │
# │ start_motors  │ start_stack (full height) │
# ├───────────────┼                           │
# │ pane 1        │                           │
# │ start_monitor │                           │
# ├───────────────┼                           │
# │ pane 2        │                           │
# │ yamcs_start   │                           │
# └───────────────┴───────────────────────────┘

# split the window
tmux split-window -h
tmux select-pane -t 0
tmux split-window -v
tmux select-pane -t 1
tmux split-window -v

tmux send-keys -t 0 'export ROS_DOMAIN_ID='$ROS_DOMAIN_ID C-m
tmux send-keys -t 1 'export ROS_DOMAIN_ID='$ROS_DOMAIN_ID C-m
tmux send-keys -t 2 'export ROS_DOMAIN_ID='$ROS_DOMAIN_ID C-m
tmux send-keys -t 3 'export ROS_DOMAIN_ID='$ROS_DOMAIN_ID C-m

# launch the scripts in each pane
tmux send-keys -t 0 './scripts/start_motors.sh' C-m
tmux send-keys -t 1 './scripts/start_monitor.sh' C-m 
# tmux send-keys -t 2 'bash ./scripts/yamcs_start.bash' C-m
tmux send-keys -t 2 "bash ${SCRIPT_DIR}/scripts/zenoh/local_server.bash" C-m
tmux send-keys -t 3 'bash ./scripts/start_stack.bash' C-m

# finally attach to the session
tmux select-pane -t 3
tmux attach -t moonbot
