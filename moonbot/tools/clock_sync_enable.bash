#!/bin/bash
set -e -o pipefail

# 1. Install chrony
# sudo apt update
sudo apt install -y chrony

# 2. Disable any other time synchronization daemons

echo ""
echo "Errors can/should happen here"
# Stop and disable ntpd (if installed)
sudo systemctl stop ntp || true
sudo systemctl disable ntp || true

# Stop and disable systemd-timesyncd (default on Ubuntu)
sudo systemctl stop systemd-timesyncd || true
sudo systemctl disable systemd-timesyncd || true
echo "Errors should end here"
echo ""

# 3. Enable and start chronyd
sudo systemctl enable chrony
sudo systemctl start chrony

# Replace or add pi55.lan
sudo sed -i '/^server\s\+pi55\.lan/d' /etc/chrony/chrony.conf
echo 'server pi55.lan iburst minpoll 2 maxpoll 4' | sudo tee -a /etc/chrony/chrony.conf

# Replace or add pi56.lan
sudo sed -i '/^server\s\+pi56\.lan/d' /etc/chrony/chrony.conf
echo 'server pi56.lan iburst minpoll 2 maxpoll 4' | sudo tee -a /etc/chrony/chrony.conf


# sudo systemctl enable chrony
sudo systemctl restart chrony
# sudo chronyc -a makestep

echo "Monitoring chrony sync status. Press Ctrl+C to stop."
while true; do
    echo "----- $(date) -----"
    chronyc sources -v | grep -E '^MS|^[\^\=].*'
    chronyc tracking | grep -E 'Stratum|Reference time|Leap status|Last offset'
    echo ""
    echo "Press Ctrl+C to exit once status is Normal. Then run 'sudo chronyc -a makestep'"
    sleep 3
done
