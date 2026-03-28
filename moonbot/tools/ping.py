import os
import re
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"

TIMEOUT = 0.5
# List of IP addresses to ping
ip_addresses = [
    "8.8.8.8",
    "192.168.8.01",
    "192.168.8.02",
    "192.168.8.03",
    "192.168.8.04",
    "192.168.8.11",
    "192.168.8.12",
    "192.168.8.13",
    "192.168.8.14",
    "192.168.8.23",
    "192.168.8.78",
]


# Function to clear terminal
def clear_terminal():
    # Windows
    if platform.system() == "Windows":
        os.system("cls")
    # Unix-based (Linux, macOS)
    else:
        os.system("clear")


# Function to ping an IP address
def ping_ip(ip):
    # # Ping parameters: "-c 1" for Unix, "-n 1" for Windows (send 1 ping)
    # param = "-n" if platform.system().lower() == "windows" else "-c"

    if platform.system().lower() == "windows":
        param = [
            "-n",
            "1",
            "-w",
            str(int(TIMEOUT * 1000)),
        ]  # Timeout in milliseconds for Windows
    else:
        param = [
            "-c",
            "1",
            "-W",
            str((TIMEOUT)),
        ]  # Timeout in seconds for Unix-based systems

    try:
        # Run the ping command
        result = subprocess.run(
            ["ping"] + param + [ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # If return code is 0, the ping was successful
        if result.returncode == 0:
            output = result.stdout.decode()
            match = re.search(r"time=([\d.]+) ms", output)
            if match:
                return (ip, f"{GREEN}{match.group(1)} ms{RESET}")  # Return the delay in ms
            else:
                return (ip, "No delay info")
        else:
            return (ip, f"{RED}Offline{RESET}")
    except Exception as e:
        return (ip, f"{RED}Error: {e}{RESET}")


# Function to ping all IPs using multithreading
def ping_all_ips(ip_list):
    with ThreadPoolExecutor() as executor:
        # Submit ping tasks to the thread pool
        results = list(executor.map(ping_ip, ip_list))
    return results


count = 0
results = dict([(ip, "In progress") for ip in ip_addresses])
results = dict(sorted(results.items()))
for ip, status in results.items():
    print(f"{ip}: {status}")
while True:
    count += 1
    count %= 3
    # Ping all IPs concurrently and get results
    r = dict(ping_all_ips(ip_addresses))
    results = dict(sorted(r.items()))

    # Display the results
    # Clear terminal before displaying the new results
    clear_terminal()
    print("Ping Results:", "." * (count + 1))
    for ip, status in results.items():
        print(f"{ip}: {status}")

    # Wait before updating (to avoid overwhelming the system with pings)
    time.sleep(0.1)
