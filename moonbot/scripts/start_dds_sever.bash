#!/bin/bash
set -eo pipefail

: ${THIS_FILE_FOLDER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"} # assumes this file location is the gusta_ws
. /opt/ros/jazzy/setup.bash || . /opt/ros/humble/setup.bash || . /opt/ros/foxy/setup.bash || echo "Could not source ros."
# . "${THIS_FILE_FOLDER}/source.bash"
. "${THIS_FILE_FOLDER}/source_fastdds.bash"


# fastdds discovery -i 0 -p 11811

if [[ "$(hostname)" == "pi55" ]]; then
    fastdds discovery -i 0 -p 11811
elif [[ "$(hostname)" == "pi56" ]]; then
    fastdds discovery -i 1 -p  11888
else
    echo "NOT LAUNCHED, this script only runs on pi55 or pi56.
    Not on the same network as the robots? start a local server using 'fastdds discovery -i 2 -p  11800 -b  -l localhost'"
fi




