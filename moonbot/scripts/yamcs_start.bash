#!/bin/bash
## This script starts the Yamcs Gateway on the robot.
## Usage: $(basename "$0") [first_setup]
##   first_setup  Install dependencies and build before starting

set -e

# Always start the gateway
echo "Starting Yamcs Gateway..."
export RUST_LOG=info
bash "$(dirname "$(readlink -f "$0")")/../cache/yamcs_moonshot/robot_start.bash"
