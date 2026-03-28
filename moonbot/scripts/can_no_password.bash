#!/bin/bash

set -e

# Check for sudo/root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo:"
    echo "   sudo bash $0"
    exit 1
fi

USER_NAME="${SUDO_USER:-$(logname)}"
RULE="$USER_NAME ALL=(ALL) NOPASSWD: /sbin/ip"

# Check if rule already exists
if sudo grep -qF "$RULE" /etc/sudoers; then
    echo "Rule already exists in sudoers."
else
    echo "Adding rule to sudoers for user $USER_NAME..."

    TMP_FILE=$(mktemp)
    sudo cp /etc/sudoers "$TMP_FILE"
    echo "$RULE" | sudo tee -a "$TMP_FILE" > /dev/null

    # Validate with visudo and then move back
    sudo visudo -c -f "$TMP_FILE"
    if [ $? -eq 0 ]; then
        sudo cp "$TMP_FILE" /etc/sudoers
        echo "Rule added successfully."
    else
        echo "Error: sudoers file is invalid. Aborting."
    fi

    rm -f "$TMP_FILE"
fi

