#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export RMW_FASTRTPS_PUBLICATION_MODE=ASYNCHRONOUS
export RMW_FASTRTPS_USE_QOS_FROM_XML=1


export FASTRTPS_DEFAULT_PROFILES_FILE="${SCRIPT_DIR}/dds_profiles/super_client.xml"

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
