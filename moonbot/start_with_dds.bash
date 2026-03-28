#!/bin/bash

: ${THIS_FILE_FOLDER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"} # assumes this file location is the gusta_ws

. "$THIS_FILE_FOLDER/source_dds.bash"
bash "$THIS_FILE_FOLDER//start_tmuxed.sh"
