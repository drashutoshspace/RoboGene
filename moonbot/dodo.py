import os
import shutil
import subprocess
from time import time
from typing import Callable, List, Union

from colorama import Back, Fore, Style
from doit.loader import create_after
from doit.task import clean_targets
from doit.tools import Interactive, PythonInteractiveAction

from config.doit_config import (
    CONFIG,
    EPOS_ARCHITECHTURE_PATH,
    GITHUB,
    INSTALL_INFERENCE,
    PATH,
    PYTHON_PKGS_TO_SKIP,
    copy_result,
    get_files,
    ros2,
    ubuntu_ver,
)

here = "./"  # working directory
yes = (
    " -y" if CONFIG.auto else ""
)  # insert this in strings to enable skipping user input


def no_ros():
    """
    Raises:
        Exception: If ros is sourced
    """
    if os.getenv("ROS_DISTRO") is not None:
        raise Exception(
            "\nError: ROS must not be sourced. Please restart in a new terminal.\n"
        )


def is_chrony_active():
    try:
        # Run systemctl is-active chrony
        result = subprocess.run(
            ["systemctl", "is-active", "chrony"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() == "active"
    except subprocess.CalledProcessError:
        return False


def make_interactive(cmd_list: List[str]):
    """Fixes interactive terminals where user input is required"""
    return [Interactive(cmd) for cmd in cmd_list]


# def task_install_canutils():
#     def is_slcand_installed():
#         import shutil
#
#         return shutil.which("slcand") is not None
#
#     return {
#         "actions": [no_ros]
#         + make_interactive(
#             [
#                 rf"sudo apt install{yes} can-utils",
#             ]
#         ),
#         "uptodate": [is_slcand_installed],
#         "verbosity": 0,
#         "doc": "Installs slcand can utilities",
#     }


def task_download():
    yield {
        "name": None,
        "actions": None,
        "doc": "Downloads and updates repositories. This must be executed alone before any other other commands to initialise dependencies.",
    }
    for repo in GITHUB.ros2_ws_pkgs:
        yield {
            "name": repo.dir_name + ":clone",
            "actions": (
                make_interactive(
                    [
                        f"git clone {GITHUB.git_options}{repo.git_link} {repo.clone_dir} -b {repo.git_branch} || true"
                    ]
                )
            ),
            "uptodate": [os.path.exists(repo.clone_dir)],
            # "verbosity": 2,
        }
        yield {
            "name": repo.dir_name + ":pull",
            "actions": [
                f"cd {repo.clone_dir} && git checkout {repo.git_branch} && git pull || true",
            ],
            "task_dep": [f"download:{repo.dir_name}:clone"],
            # "uptodate": repo.uptodate,
            # "verbosity": 2,
        }


def task_workspace():
    yield {
        "name": None,
        "actions": None,
        "doc": "",
    }
    yield {
        "name": "file structure",
        "actions": [f"mkdir -p {os.path.join(PATH.workspace, 'src')}"],
        "uptodate": [os.path.exists(os.path.join(PATH.workspace, "src"))],
    }
    yield {
        "name": f"symlink repo -> cache",
        "actions": [
            lambda *_: os.symlink(
                src=f"{here}/repos",
                dst=f"{here}/cache",
                target_is_directory=True,
            ),
        ],
        "uptodate": [os.path.exists(f"{here}/cache")],
    }
    for repo in GITHUB.ros2_ws_pkgs:
        for pkg_rel, pkg_abs in zip(
            repo.packages_rel_sources, repo.pacakges_abs_sources
        ):
            pkg_rel = os.path.split(os.path.abspath(pkg_abs).removesuffix("/."))[-1]
            dst = os.path.abspath(
                os.path.join(
                    repo.pacakges_abs_destination.removesuffix("/."),
                    pkg_rel.removesuffix("/."),
                ).removesuffix("/.")
            )
            src_dir = os.path.abspath(pkg_abs.removesuffix("/."))
            yield {
                "name": f"symlink {src_dir} -> {dst}",
                "actions": [
                    f"mkdir -p {repo.pacakges_abs_destination.removesuffix('/.')}",
                    lambda *_, src_dir=src_dir, dst=dst: os.symlink(
                        src=src_dir,
                        dst=dst,
                        target_is_directory=True,
                    ),
                ],
                "uptodate": [os.path.exists(dst)],
            }


def task_limb_id():
    env_var = "LIMB_ID"

    def user_input():
        mleg: str = input(
            f"{Fore.RED}{env_var} is not set in ~/.bashrc.{Fore.RESET} {env_var} identifies what\n\
robot the code is running on. This ID follows the style\n\
<brand>-<module>-<version>-<number>, or can be `ALL` for a\n\
workstation that is not the robot. \n\
\n\
Here is the list of supported brands, modules and versions as examples:\n\
hero-arm-v1-X: Hero Arm v1 number X \n\
hero-arm-v2-X: Hero Arm v2 number X \n\
hero-wheel-v0-X: Hero Wheel v0 (2 wheel joints) number X \n\
hero-wheel-v1-X: Hero Wheel v1 (2 wheel joints + grip joint) number X \n\
hero-wheel-v2-X: Hero Wheel v2 (2 wheel joints + 2 flipper) number X \n\
gustavo-arm-v3-X: Gustavo Arm v3 (4 dof + grippers) number X \n\
ALL: Workstation for simulation and developpment\n\
{Fore.CYAN}Input the relevant {env_var}:{Fore.RESET} "
        )
        with open(os.path.expanduser("~/.bashrc"), "a") as f:
            f.write(f"export {env_var}={mleg}\n")
        print(
            f"Added `export {env_var}={mleg}` to ~/.bashrc.\n!!! IMPORTANT !!! new terminal needed to take effect."
        )

    return {
        "actions": ([PythonInteractiveAction(user_input)]),
        "uptodate": [os.getenv(env_var) is not None and os.getenv(env_var) != ""],
        "doc": "Forces the user to input leg number",
        # "verbosity": 2,
    }


def task_install_epos():
    return
    return {
        "actions": (
            [
                f"cd {GITHUB.hero_epos.clone_dir}/EPOS_Linux_Library/ && sudo bash {PATH.hero_epos}/EPOS_Linux_Library/install.sh",
                f"cd {GITHUB.hero_epos.clone_dir}/EPOS_Linux_Library/lib{EPOS_ARCHITECHTURE_PATH} && sudo cp libEposCmd.so.6.8.1.0 /usr/local/lib",
                f"cd /usr/local/lib && sudo ln -s -f libEposCmd.so.6.8.1.0 libEposCmd.so",
                f"cd /usr/lib && sudo ln -s -f /usr/local/lib/libEposCmd.so libEposCmd.so",
                # I don't like the last step
            ]
        ),
        "targets": [
            "/usr/local/lib/libEposCmd.so.6.8.1.0",
            "/usr/local/lib/libEposCmd.so",
            "/usr/lib/libEposCmd.so",
        ],
        "uptodate": [
            os.path.exists("/usr/local/lib/libEposCmd.so.6.8.1.0"),
            os.path.exists("/usr/local/lib/libEposCmd.so"),
            os.path.exists("/usr/lib/libEposCmd.so"),
        ],
        # "clean": False,
        "doc": "installs the epos library",
        # "verbosity": 2,
    }


def remove_dir(dirs: Union[List[str], str]) -> List[Callable]:
    """recursively removes a directory

    The list of callable can then be passed to the `clean` field of a task"""
    if isinstance(dirs, str):
        dir_list = [dirs]
    else:
        dir_list = dirs
    out: List[Callable] = []
    for d in dir_list:
        if os.path.exists(d):

            def f(x=d):
                if not os.path.exists(x):
                    return
                try:
                    if os.path.islink(x) or os.path.isfile(x):
                        os.remove(x)
                    else:
                        shutil.rmtree(x)
                except PermissionError:
                    print(
                        f"Warning: [PermissionError] deleting file ´{x}´\nUse sudo doit clean?"
                    )

            out.append(f)
    #
    return out
