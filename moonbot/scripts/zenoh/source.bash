#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

export ZENOH_ROUTER_CONFIG_URI="${SCRIPT_DIR}/profile/local_router.json5"
export ZENOH_SESSION_CONFIG_URI="${SCRIPT_DIR}/profile/client.json5"
