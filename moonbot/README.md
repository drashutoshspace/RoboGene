# MoonBot

Tools for running RoboGene morphologies on MoonBot hardware.

| Path | Content |
| --- | --- |
| `replay_commands.py`, `replay_vehicle.yaml` | Replays a `commands.csv` from `python -m robogene.sim2real` on the robot in open loop, publishing `sensor_msgs/JointState` on `/leg{N}/joint_set` of the motion stack |
| `dcf_epos_studio/` | Drive configurations of the Wheel and Limb module motors |
| `dodo.py`, `config/`, `pixi.toml`, `pixi.lock` | Environment and ROS 2 workspace build (`pixi run download`, then `pixi run build`) |
| `start_tmuxed.sh`, `operator.bash`, `scripts/` | Starting the motors, the motion stack, the operator and communication (DDS, Zenoh) |
| `tools/` | Workspace sourcing, URDF export (`urdf_extract.bash`), clock sync, sensors |

Private repositories are downloaded with SSH keys, or over HTTPS with a token in the
`GITHUB_TOKEN` environment variable.

## Replaying a command sequence

1. Export the commands of a trained morphology:
   `python -m robogene.sim2real --entry <run>/individuals/<id>/robogene.json --mapping configs/sim2real_vehicle.yaml --out commands.csv`
2. Set the joint names in `replay_vehicle.yaml` to those of the robot URDF
   (`bash tools/urdf_extract.bash`); every `SET_` placeholder must be replaced.
3. Check the messages without ROS: `python replay_commands.py --commands commands.csv --dry-run`
4. With the motion stack running on the robot: `python replay_commands.py --commands commands.csv`

Wheel commands are velocities (rad/s), limb commands are joint positions (rad). The replay
sends zero wheel velocity when it ends or is interrupted.
