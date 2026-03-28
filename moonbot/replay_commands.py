"""Replay a RoboGene command sequence on MoonBot through the motion stack (open loop)."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Channel:
    column: str
    leg: int
    joint: str
    field: str  # "velocity" or "position"


def load_channels(path: str | Path) -> list[Channel]:
    """Read the column -> (leg, joint, field) mapping."""
    cfg = yaml.safe_load(Path(path).read_text())
    channels = [Channel(c["column"], int(c["leg"]), str(c["joint"]), c["field"]) for c in cfg["channels"]]
    for c in channels:
        if c.field not in ("velocity", "position"):
            raise ValueError(f"{c.column}: field must be velocity or position")
        if not c.joint or c.joint.startswith("SET_"):
            raise ValueError(f"{c.column}: set the joint name from the robot URDF")
    return channels


def load_commands(path: str | Path) -> tuple[list[float], dict[str, list[float]]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    t = [float(r["t"]) for r in rows]
    columns = {k: [float(r[k]) for r in rows] for k in rows[0] if k != "t"}
    return t, columns


def messages(t: list[float], columns: dict[str, list[float]], channels: list[Channel]):
    """Yield (time, {leg: (names, positions, velocities)}) for every command step."""
    missing = [c.column for c in channels if c.column not in columns]
    if missing:
        raise ValueError(f"columns not in the command file: {missing}")
    for i, ti in enumerate(t):
        per_leg: dict[int, tuple[list[str], list[float], list[float]]] = {}
        for c in channels:
            names, pos, vel = per_leg.setdefault(c.leg, ([], [], []))
            names.append(c.joint)
            (vel if c.field == "velocity" else pos).append(columns[c.column][i])
        yield ti, per_leg


def stop_messages(channels: list[Channel]) -> dict[int, tuple[list[str], list[float], list[float]]]:
    """Zero velocity for every velocity-controlled joint."""
    per_leg: dict[int, tuple[list[str], list[float], list[float]]] = {}
    for c in channels:
        if c.field == "velocity":
            names, _, vel = per_leg.setdefault(c.leg, ([], [], []))
            names.append(c.joint)
            vel.append(0.0)
    return per_leg


def _valid(names, pos, vel) -> bool:
    # a JointState carries either positions or velocities for all of its names
    return not (pos and vel)


def replay(commands: str, mapping: str, rate_scale: float = 1.0, dry_run: bool = False) -> None:
    channels = load_channels(mapping)
    t, columns = load_commands(commands)
    steps = list(messages(t, columns, channels))
    for _, per_leg in steps[:1]:
        for leg, msg in per_leg.items():
            if not _valid(*msg):
                raise ValueError(f"leg {leg} mixes position and velocity joints; use one field per leg")
    if dry_run:
        for ti, per_leg in steps[:5]:
            for leg, (names, pos, vel) in sorted(per_leg.items()):
                print(f"t={ti:.2f} /leg{leg}/joint_set names={names} position={pos} velocity={vel}")
        print(f"{len(steps)} steps, {t[-1] - t[0]:.2f} s")
        return

    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("robogene_replay")
    legs = sorted({c.leg for c in channels})
    pubs = {leg: node.create_publisher(JointState, f"/leg{leg}/joint_set", 10) for leg in legs}

    def publish(per_leg):
        for leg, (names, pos, vel) in per_leg.items():
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name, msg.position, msg.velocity = names, pos, vel
            pubs[leg].publish(msg)

    start = time.monotonic()
    try:
        for ti, per_leg in steps:
            delay = start + (ti - t[0]) / rate_scale - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            publish(per_leg)
    finally:
        publish(stop_messages(channels))
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commands", required=True, help="commands.csv from robogene.sim2real")
    p.add_argument("--mapping", default=str(Path(__file__).with_name("replay_vehicle.yaml")))
    p.add_argument("--rate-scale", type=float, default=1.0, help="replay speed factor")
    p.add_argument("--dry-run", action="store_true", help="print the first messages without ROS")
    a = p.parse_args(argv)
    replay(a.commands, a.mapping, a.rate_scale, a.dry_run)


if __name__ == "__main__":
    main()
