import dataclasses
import os
import platform
import sys
from glob import glob
from typing import List, Literal, Optional

import doit


class CONFIG:
    # if cli argument has something else than`auto=n`, user interaction prompts will be displayed
    auto = doit.get_var("auto", "y") != "n"


# Check if user wants to install rl_inference package (requires SSH access)
# Set INSTALL_INFERENCE=True in environment to enable rl inference pkg if you dont have isaac lab then dont do True
INSTALL_INFERENCE = os.getenv("INSTALL_INFERENCE", "False").lower() == "true"


# if MBT_GIT_SSH is set to 1 in the OS env var, then we use ssh
mbt_git_ssh = os.getenv("MBT_GIT_SSH", "0")
GIT_STRATEGY: Literal["https", "ssh"]

# method use to download repositories
if mbt_git_ssh == "1":
    GIT_STRATEGY = "ssh"
else:
    GIT_STRATEGY = "https"

if any(arch in platform.machine() for arch in ["x86_64"]):
    EPOS_ARCHITECHTURE_PATH = "/intel/x86_64"
elif any(arch in platform.machine() for arch in ["i386", "i686"]):
    EPOS_ARCHITECHTURE_PATH = "/intel/x86"
elif any(arch in platform.machine() for arch in ["armv8", "aarch64"]):
    EPOS_ARCHITECHTURE_PATH = "/arm/v8"  # this is for rasberry pi
elif any(arch in platform.machine() for arch in ["armv7"]):
    EPOS_ARCHITECHTURE_PATH = "/arm/v7"
elif any(arch in platform.machine() for arch in ["armv6"]):
    EPOS_ARCHITECHTURE_PATH = "/arm/v6"
else:
    raise ValueError("CPU architechture not supported by the Epos library")

ubuntu_ver = platform.freedesktop_os_release()[
    "VERSION_ID"
]  # WARNING: not available on 20.04 / foxy
ros2: Literal["foxy", "humble", "jazzy"] = {  # type: ignore
    "20.04": "foxy",
    "22.04": "humble",
    "24.04": "jazzy",
}[ubuntu_ver]


class PATH:
    """Paths and file locations"""

    # location of this folder
    setup = os.path.abspath("./")

    # Everything will be installed here
    # root = os.path.expanduser("~/")
    root = setup

    # full github repos are cached in here.
    # use doit download to create/refresh the cache.
    cache = os.path.join(setup, "repos")
    # If you think that this doit is a build-system (it kinda is, but a shitty one),
    # this 'cache' is the source used to build everything.
    # this way, one can rsync or symlink its work in there, and everything will build easily
    # we can also replace gitpull with a local server to refresh the cache

    # motion stack and everything will be copied in hero_workspace/src
    workspace = os.path.join(root, "workspace")
    hero_ros2 = os.path.join(workspace, "src", "hero_ros2")
    monitor = os.path.join(workspace, "src", "monitor")
    realman_src = os.path.join(workspace, "src")
    launcher = os.path.join(workspace, "src", "moonbot_launcher")
    hero_epos = os.path.join(workspace, "src", "hero_epos")
    odrive = os.path.join(workspace, "src", "ros_odrive")
    gusta_pkg = os.path.join(workspace, "src", "moonbotg_4dof_rppr_ultraslim")
    ms_src = os.path.join(workspace, "src")
    gleg_commander = os.path.join(workspace, "src", "gleg_commander")
    
    # Conditional path - only if INSTALL_INFERENCE is True
    if INSTALL_INFERENCE:
        rl_inference = os.path.join(workspace, "src", "rl_inference")

    # setup = os.path.abspath("~/setup/")
    ros2_setup = f"/opt/ros/{ros2}/setup.bash"


def get_files_reg(path_regex: str) -> List[str]:
    dirs_and_files = glob(path_regex, recursive=True)
    return [f for f in dirs_and_files if os.path.isfile(f)]


def get_files(path: str) -> List[str]:
    """Gets all files in a directory and its children dirs"""
    return get_files_reg(os.path.join(path, "**"))


def copy_result(src: str, dst: str, files: Optional[List[str]] = None) -> List[str]:
    """Returns all files that would be created by copying the src directory.

    Args:
        files: manually input files
    """
    if files is None:
        files = get_files(src)
    result: List[str] = []
    head = dst
    for f in files:
        tail = os.path.relpath(f, src)
        result.append(os.path.join(head, tail))
    return result


@dataclasses.dataclass
class GitRepo:
    """Class to help caching github repos"""

    dir_name: str
    git_link: str
    packages_rel_sources: List[str] = dataclasses.field(
        default_factory=lambda *_: ["."]
    )
    packages_rel_destination: str = "."
    git_branch: str = "HEAD"

    @property
    def clone_dir(self) -> str:
        """directory of the cached repo"""
        return os.path.join(PATH.cache, self.dir_name)

    @property
    def pacakges_abs_sources(self) -> List[str]:
        """directory of the cached repo"""
        return [os.path.join(self.clone_dir, p) for p in self.packages_rel_sources]

    @property
    def pacakges_abs_destination(self) -> str:
        """directory of the cached repo"""
        return os.path.join(PATH.workspace, "src")

    @property
    def cached_files(self) -> List[str]:
        """Targets indicating the cached repo files.

        As long as the repo is not downloaded this will be empty"""
        return get_files(self.clone_dir)

    @property
    def uptodate(self) -> List[str]:
        """Shell Command returning false if the cache is out of date

        use this in doit task's up_to_date field."""
        return [
            f"""cd {self.clone_dir} && git fetch && [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/{self.git_branch})" ]"""
        ]


class GITHUB:
    """relevant github repos links and stuff"""

    # https clones read GITHUB_TOKEN through a credential helper, so the token is never stored in a URL
    _srl_pat_prefix = "https://github.com/"
    _token_helper = (
        "-c credential.helper='!f() { echo username=x-access-token; echo password=$GITHUB_TOKEN; }; f' "
        if os.environ.get("GITHUB_TOKEN")
        else ""
    )

    _ssh_prefix = f"git@github.com:"

    # git_options = "--depth 1 " # use this to download faster
    git_options = _token_helper

    _strategy_prefix = {
        "https": _srl_pat_prefix,
        "ssh": _ssh_prefix,
    }[GIT_STRATEGY]

    launcher = GitRepo(
        dir_name="moonbot_launcher",
        git_link=f"{_strategy_prefix}Space-Robotics-Laboratory/"
        + "moonbot_launcher.git",
        git_branch="main",
    )
    ms = GitRepo(
        dir_name="motion_stack",
        git_link=f"{_strategy_prefix}2lian/Motion-Stack.git",
        git_branch="main",
        packages_rel_sources=[
            "./src/ms_operator",
            "./src/sim",
            # VVV We have binaries
            # "./src/moonbot_zero_tuto",
            # "./src/motion_stack",
            # "./src/motion_stack_msgs",
            # "./src/keyboard_event/",
            # "./src/keyboard_event_msgs/",
        ],
    )
    hero_ros2 = GitRepo(
        dir_name="hero_ros2",
        git_link=f"{_strategy_prefix}Space-Robotics-Laboratory/" + "hero_ros2.git",
        git_branch="main",
    )
    monitor = GitRepo(
        dir_name="monitor",
        git_link=f"{_strategy_prefix}Space-Robotics-Laboratory/"
        + "moonbot_hero_monitor_ros2.git",
        git_branch="develop",
    )
    hero_epos = GitRepo(
        dir_name="hero_epos",
        git_link=f"{_strategy_prefix}Space-Robotics-Laboratory/" + "hero_epos.git",
        git_branch="pixi",
    )
    odrive = GitRepo(
        dir_name="ros_odrive",
        git_link=f"{_strategy_prefix}2lian/" + "ros_odrive.git",
        git_branch="pixi",
        packages_rel_sources=[
            "./odrive_base",
            "./odrive_node",
        ],
    )
    gleg_commander = GitRepo(
        dir_name="gleg_commander",
        git_link=f"{_strategy_prefix}Space-Robotics-Laboratory/" + "gleg_commander.git",
        git_branch="main",
    )
    
    # Conditional repository - only available if user sets INSTALL_INFERENCE=True
    rl_inference = GitRepo(
        dir_name="rl_inference",
        git_link=f"{_ssh_prefix}drashutoshspace/" + "rl_inference.git",  # Always use SSH for this repo
        git_branch="main",
    ) if INSTALL_INFERENCE else None
    
    gusta = GitRepo(
        dir_name="gustaws",
        git_link=f"{_strategy_prefix}Space-Robotics-Laboratory/"
        + "moonbotg_4dof_rppr_ultraslim.git",
        git_branch="main",
        packages_rel_sources=[
            "./src/moonbotg_4dof_rppr_ultraslim",
        ],
    )

    # keyboard = GitRepo(
    #     # we pre-download this one, otherwise the motion stack will doit
    #     dir_name=f"{ms.clone_dir}/src/ros2-keyboard",
    #     git_link=f"{_strategy_prefix}cmower/ros2-keyboard",
    #     git_branch="master",
    # )

    realman = GitRepo(
        dir_name="realman_motion_stack",
        git_link=f"{_strategy_prefix}shamikarim/" + "realman_motion_stack.git",
        git_branch="main",
    )

    ros2_ws_pkgs: List[GitRepo] = [
        launcher,
        ms,
        hero_ros2,
        monitor,
        gusta,
        gleg_commander,
    ] + ([rl_inference] if rl_inference is not None else [])


PYTHON_PKGS_TO_SKIP = [
    "matplotlib"
]  # there is a problem on te university network, we cannot download matplotlib, so we skip it. It is not needed anyway because it's part of ros2.
