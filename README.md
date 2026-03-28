# RoboGene

RoboGene evolves the morphology of modular reconfigurable robots for a given task. Every
module is described by a standardized digital entry (a RoboGene) and every robot by a
genotype graph of modules joined at their connection ports. A genetic algorithm searches
over genotype graphs; each candidate is built as a MuJoCo model under lunar gravity, a
control policy is trained for it with PPO, and the progress of the trained policy on the
task, in meters, is its fitness.

## Layout

| Part | Code |
| --- | --- |
| RoboGene entries and module library | `robogene/schema.py`, `robogene/library/` |
| Genotype graph, port alignment, validation, morphology classes | `robogene/genotype.py` |
| Phenotype (MuJoCo model) and its checks, terrain | `robogene/phenotype.py`, `robogene/terrain.py` |
| Locomotion and loco-manipulation environments | `robogene/envs.py` |
| PPO training and fitness evaluation | `robogene/rl.py` |
| Genetic algorithm, evolution loop, run archive | `robogene/ga.py`, `robogene/evolve.py`, `robogene/archive.py` |
| Terrain robustness evaluation | `robogene/terrain_eval.py` |
| Figures | `robogene/plots.py` |
| Hardware command export for MoonBot | `robogene/sim2real.py` |
| MoonBot hardware tools and command replay | `moonbot/` |
| Examples | `examples/` |

Interfaces between the parts are described in [`docs/architecture.md`](docs/architecture.md).

## Installation

Python 3.10 to 3.12.

```sh
uv venv --python 3.12 .venv
uv pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU build, optional
uv pip install -e ".[dev]"
```

`pip install -e ".[dev]"` inside any virtual environment works as well.

## Running

A tiny run that exercises the whole pipeline in a few minutes:

```sh
python -m robogene.evolve --config configs/smoke.yaml --run-name smoke
```

Locomotion toward a random goal, and locomotion with arm reaching:

```sh
python -m robogene.evolve --config configs/task1_locomotion.yaml --run-name task1 --workers 16
python -m robogene.evolve --config configs/task2_loco_manip.yaml --run-name task2 --workers 16
```

With 16 candidates per generation, each trained for 2 x 10^5 steps in every generation, a
99-generation run comprises 1584 PPO trainings and a 56-generation run 896. `--workers`
trains that many candidates at the same time, one process each.

Terrain robustness (policies trained on flat terrain, run on hilly terrain without retraining):

```sh
python -m robogene.terrain_eval --runs runs/task1 runs/task2 --config configs/terrain_hilly.yaml --out runs/terrain.csv
```

Figures:

```sh
python -m robogene.plots generations --run runs/task1
python -m robogene.plots boxplot --run runs/task1
python -m robogene.plots tradeoff --run runs/task2
python -m robogene.plots terrain --csv runs/terrain.csv
```

Hardware command sequence for MoonBot (wheel actions recorded from the trained policy in
simulation, for open-loop replay on the wheel modules, and a constant limb command from the
center arm):

```sh
python -m robogene.sim2real --entry runs/task1/individuals/g098_i00/robogene.json --mapping configs/sim2real_vehicle.yaml --out commands.csv
```

Edit `configs/sim2real_vehicle.yaml` so that its node ids match the chosen morphology.

## Examples

Run from the repository root:

```sh
python examples/build_phenotype.py --render        # genotype -> validated MuJoCo phenotype (MUJOCO_GL=egl when headless)
python examples/train_policy.py --task loco_manip  # PPO for one morphology and its fitness
python examples/evolve_small.py                     # a short evolution with its figures
python examples/moonbot_commands.py                 # train a rover, export MoonBot commands, preview the replay
```

## MoonBot

`moonbot/` holds the drive configurations, robot stack scripts and a replay tool that sends an
exported command sequence to MoonBot through the motion stack. See
[`moonbot/README.md`](moonbot/README.md).

## Configuration

Every run is described by one YAML file in `configs/`. Keys that are not given take the
defaults in `robogene/config.py`, which also documents every setting: simulation (1 kHz
physics, lunar gravity 1.62 m/s^2, 50 Hz control), terrain, episode length and number of
evaluation episodes, goal sampling, reward coefficients, GA settings (M, K, G, tournament
size, crossover and mutation probabilities), validity checks and PPO settings.

## Outputs

```
runs/<run_name>/
  config.yaml                 the merged configuration of the run
  evolution_log.csv           one row per evaluated candidate
  best_per_generation.csv
  final_top_k.json            the K best genotypes of the run
  individuals/g{gen}_i{idx}/  robogene.json (robot-level RoboGene entry), policy.zip, model.xml
```

## Module library

`robogene/library/modules/` holds the simplified modules used in the search (small and large
bodies, a connector, an actuated hinge, an arm segment, left and right wheels).
`robogene/library/assemblies/` holds arms of three and five segments.
`robogene/library/hardware/` describes the MoonBot Wheel and Limb modules for sim-to-real
transfer. Their velocity and torque limits come from the MoonBot motor drive configuration
(gear reduction 960:1, maximum gear input speed 6000 rpm, rated motor torque of each drive).

## Tests

```sh
python -m pytest
```

## License

MIT. See [LICENSE](LICENSE).

## About this release

This repository is the open-source version of RoboGene. It showcases the pipeline and how it works, from module
library and genotype graphs to evolution, training and evaluation. For the full code, please contact the owner of
this repository.
