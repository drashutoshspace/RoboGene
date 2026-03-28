# Architecture

Interfaces between the parts of the code.

```
Library (RoboGene entries)            robogene/schema.py, robogene/library/
  -> Genotype G = (V, E)              robogene/genotype.py
  -> Phenotype = Expand(G, RoboGene)  robogene/phenotype.py, robogene/terrain.py
  -> environments                     robogene/envs.py
  -> PPO training + fitness F         robogene/rl.py
  -> genetic algorithm                robogene/ga.py, robogene/evolve.py, robogene/archive.py
  -> analysis                         robogene/terrain_eval.py, robogene/plots.py, robogene/sim2real.py
```

Configuration is one nested dict (`robogene.config.load_config`). The keys and defaults are
in `DEFAULT_CONFIG` in `robogene/config.py`. Sections: `simulation`, `terrain`, `episode`,
`goal`, `reward`, `ga`, `checks`, `rl`, `parallel`, plus `task`, `seed`, `run_dir`.

## 1. Library and genotype

- `robogene.schema.Library`, `default_library()`. `library[name]` is a `ModuleEntry` with
  `category` in {body, connector, hinge, arm, wheel}, `geometry` (shape box / cylinder /
  capsule, MuJoCo half-sizes, cylinders and capsules along the module z axis, density),
  `actuation` (`joint` fixed / hinge / continuous, `axis` in the module frame, `anchor`
  "port" = the joint sits at the port through which the module attaches to its parent, or
  "center", `control_mode` position / velocity, `range` rad, `velocity_limit`,
  `torque_limit`, `kp`, `kv`, `damping`, `armature`), `ports` (id -> Port with `position`,
  outward unit `normal`, `roll_options_deg`), `mass()`, `inertia()` (diagonal, about the
  center), `bounding_radius()`.
- `library.assemblies["arm_3dof" | "arm_5dof"]`: chains of arm segments, attach via
  `attach_port` ("end_a") of their first segment. `library.hardware["moonbot_wheel_module" |
  "moonbot_limb"]`: real MoonBot joints with velocity and torque limits.
- `robogene.genotype`: `Genotype(nodes, edges, root)`, `Node(id, module, params)`,
  `Edge(parent, parent_port, child, child_port, roll_deg)`. Useful methods: `subtree`,
  `children`, `parent_edge`, `free_ports(nid, library)`, `add_node`, `remove_subtree`,
  `graft`, `relabel`, `canonical_hash`, `to_dict` / `from_dict`.
  Functions: `edge_transform(g, e, library) -> (R, t)` (child frame in parent frame),
  `forward_kinematics(g, library) -> {nid: (R_world, p_world)}` at zero joint angles,
  `validate_genotype(g, library, cfg["checks"], cfg["ga"]) -> (ok, reasons)`,
  `module_counts`, `arm_chains`, `morphology_class` ("3Arm+4Wheel", "Arm Only", "Legged",
  "4Wheel", "Body Only").
- Node parameter `params["limit_scale"]` in (0, 1] scales the joint range of a hinge or arm
  segment (mutation "perturb joint limits"). Default 1.0.

## 2. Phenotype (robogene/phenotype.py, robogene/terrain.py)

MJCF naming, used by every other part:

| Item | Name |
| --- | --- |
| body of node i | `n{i}` (root body carries a free joint named `root`) |
| geom of node i | `g{i}` |
| joint of node i | `j{i}` |
| actuator of node i | `a{i}` |
| arm-tip site used as end effector | `ee` |
| site at the root body origin | `base` |
| ground geom (plane or height field) | `ground` |

```python
@dataclass
class ActuatorInfo:
    name: str                 # "a{i}"
    joint: str                # "j{i}"
    node: int
    module: str
    category: str             # "wheel" | "arm" | "hinge"
    control_mode: str         # "position" | "velocity"
    ctrl_range: tuple[float, float]
    velocity_limit: float

@dataclass
class Phenotype:
    genotype: dict            # Genotype.to_dict(), keeps the object picklable
    morphology_class: str
    robot_xml: str            # the robot <body> tree only (worldbody content)
    actuator_xml: str         # <actuator> children
    actuators: list[ActuatorInfo]   # same order as MuJoCo actuators, so action i -> actuator i
    end_effector: int | None  # node id whose far end carries site "ee", None without arms
    total_mass: float         # kg
    spawn_clearance: float    # m from root origin down to the lowest point at zero joint angles
    declared_mass: dict[int, float]      # from RoboGene entries, for the mass check
    declared_inertia: dict[int, list]    # diagonal, for the inertia check

def expand(g: Genotype, library: Library | None = None) -> Phenotype       # Expand(G, RoboGene)
def model_xml(ph: Phenotype, cfg: dict, spawn_xy=(0.0, 0.0)) -> str          # full MJCF document
def build_model(ph: Phenotype, cfg: dict, spawn_xy=(0.0, 0.0)) -> mujoco.MjModel
def check_phenotype(ph: Phenotype, cfg: dict) -> tuple[bool, list[str], dict]
```

`model_xml` writes `<option timestep gravity="0 0 -g" integrator>`, the terrain from
`robogene.terrain`, the robot with its root placed so that the lowest point clears the
ground by 0.02 m at `spawn_xy`, and the actuators. Hinge-type modules get a `position`
actuator (`kp`, ctrlrange = joint range x limit_scale, forcerange = torque limit); wheels get
a `velocity` actuator (`kv`, ctrlrange = +-velocity_limit, forcerange = torque limit).
Joints get `damping` and `armature` from the entry. A module on the root has no joint.

`check_phenotype` applies the enabled `cfg["checks"]`: mass and inertia of every body in
the compiled model within `mass_inertia_tol` (0.10) of the RoboGene entry; settle for
`simulation.settle_time` with zero action, then: no self-penetration between robot bodies
that are not parent and child (`self_penetration`); `stability`: base tilt at rest <=
`max_tilt_deg` and center-of-mass projection inside the support polygon of ground contacts
with a margin > `com_margin` x polygon width; `reachability`: an end effector exists
and its arm chain is at least `reach_min` long. Returns `(ok, reasons, metrics)`.

`robogene/terrain.py`:

```python
def terrain_assets_xml(terrain_cfg: dict) -> str      # <asset> children (hfield for hilly)
def terrain_geom_xml(terrain_cfg: dict) -> str        # the "ground" geom
def hilly_heights(terrain_cfg: dict) -> np.ndarray    # (ny, nx) normalized heights in [0, 1]
def ground_height(terrain_cfg: dict, x: float, y: float) -> float   # world z of the ground
```

Flat: a plane named `ground` with `friction` from the config. Hilly: MuJoCo `hfield` with
nrow = ny, ncol = nx, heights drawn from N(mean, std) with `seed` and clipped to [0, 1]
(rows written from +y to -y). MuJoCo rescales elevation data by its min and max, so the
field has vertical size max_height x (hmax - hmin) and sits at z = (hmin - mean) x max_height,
which makes the surface z = (h - mean) x max_height. A plane `ground_base` at the field's
lowest level continues the world outside it. Ground geoms have priority 1 so their friction
is used in contacts. `max_ground_height(terrain_cfg, x_range, y_range)` gives the exact
maximum over a rectangle, used for spawning.

## 3. Environments and RL (robogene/envs.py, robogene/rl.py)

```python
class RoboGeneEnv(gymnasium.Env):
    def __init__(self, phenotype: Phenotype, cfg: dict, render_mode=None): ...
    # action: Box(-1, 1, (n_act,)), mapped linearly to each actuator ctrl_range
    # observation, scaled in the env and clipped to +-10:
    #   position-joint angles normalized to [-1, 1] over the ctrl range (n_pos)
    #   actuated joint velocities divided by the velocity limit (n_act)
    #   gravity direction in the base frame (3), base linear velocity in the base frame (3)
    #   goal vector (x, y) in the base frame and planar distance, divided by 10 (3)
    #   ground-contact flag per wheel, then one for the end-effector body if "ee" exists
    #   loco_manip only: target - end-effector vector in the base frame / 10 (3)
    # reset(seed=None, options=None): options may hold "goal": (x, y) and "target_height"
    # step: frame_skip physics steps, then
    #   delta_t = ||p_{t-1} - p_g|| - ||p_t - p_g||        (planar base position)
    #   delta_arm_t = ||e_{t-1} - e_g|| - ||e_t - e_g||    (3D end effector)
    #   locomotion: r = alpha*delta_t - beta*||a_t - a_{t-1}||^2 - delta*p_fall - zeta*p_coll
    #   loco_manip: r = alpha*delta_t + w_arm*delta_arm_t - beta*||a_t - a_{t-1}||^2 - gamma_r*||qdot||^2
    #   p_fall = 1 if the base is tilted beyond fall_tilt_deg, p_coll = 1 if any two robot
    #   bodies that are not parent and child touch.
    #   info: {"progress", "arm_progress", "fall", "collision", "base_pos", "ee_pos"}
    #   truncated after episode.length steps; terminated on numerical blow-up or (if
    #   reward.terminate_on_fall) on a fall
```

Goals: p_g uniform in `goal.x_range` x `goal.y_range`, resampled until at least
`goal.min_distance` from the spawn point. The loco_manip target e_g = (p_g, ground height at p_g +
U(target_height)). Each episode starts with `simulation.settle_time` of zero action.

```python
def make_env(ph, cfg, seed=None) -> Callable[[], RoboGeneEnv]      # picklable factory
def train_policy(ph, cfg, seed: int, log_dir=None) -> stable_baselines3.PPO
def evaluate_policy(model, ph, cfg, episodes=None, seed=None) -> dict
    # deterministic actions, N = episode.eval_episodes, goals from seeds eval_seed + k
    # returns {"loco": mean sum delta_t, "manip": mean sum delta_arm_t,
    #          "fitness": loco or manip by task, "per_episode": [...]}  (meters)
def evaluate_individual(genotype: dict, cfg: dict, out_dir: str, seed: int) -> dict
    # expand -> train -> evaluate -> save out_dir/policy.zip and out_dir/model.xml
    # returns {"fitness", "loco", "manip", "obs_dim", "act_dim", "train_time_s", "policy": "policy.zip"}
```

PPO settings come from `cfg["rl"]`: Stable-Baselines3 PPO, MlpPolicy
with net_arch [64, 64], Tanh, orthogonal init, other settings at their defaults. Every
policy is trained from scratch for `rl.budget` steps.

## 4. Genetic algorithm (robogene/ga.py, robogene/archive.py, robogene/evolve.py)

Evolution loop: sample M valid genotypes from the library; in each generation train and score
every member (elites included); archive every evaluation; keep the K best; fill the other
M - K places with offspring made by tournament selection, subgraph crossover and mutation,
keeping only offspring that pass `validate_genotype` and `check_phenotype`; after G
generations return the K best evaluated genotypes in the archive.

Mutations: substitute a module by another of the same category, reassign a subtree to
another free port of the same parent or change its roll, add a leaf (a module or an arm
assembly) at a free port, remove a leaf, perturb `limit_scale`.

Run directory layout:

```
runs/<run_name>/
  config.yaml
  evolution_log.csv        generation,index,id,hash,morphology_class,n_modules,arms,wheels,
                           fitness,loco,manip,train_time_s,elite
  best_per_generation.csv  generation,id,morphology_class,fitness
  final_top_k.json
  individuals/g{gen:03d}_i{idx:02d}/robogene.json, policy.zip, model.xml
```

`robogene.json` is a robot-level RoboGene entry with these fields:

| Field | Content |
| --- | --- |
| `robogene_version`, `type` | "1.0", "robot" |
| `id`, `generation`, `index` | e.g. "g003_i07", 3, 7 |
| `task` | "locomotion" or "loco_manip" |
| `genotype`, `hash`, `morphology_class` | `Genotype.to_dict()`, `canonical_hash()`, class name |
| `modules` | module counts per category and in total |
| `fitness` | `value`, `loco`, `manip` (meters), `episodes`, `unit` |
| `policy` | `checkpoint`, `obs_dim`, `act_dim`, `control_hz`, `normalization` |
| `simulation` | `mjcf`, `gravity`, `timestep`, `terrain` |

## 5. Analysis

- `python -m robogene.terrain_eval --runs <run> [<run> ...] --config configs/terrain_hilly.yaml`
  evaluates each unique morphology (by hash, best flat entry) on hilly terrain without
  retraining and writes `terrain.csv` (id, run, hash, morphology_class, flat_fitness,
  hilly_fitness, ratio).
- `python -m robogene.plots generations|boxplot|tradeoff|terrain ...` draws the result
  figures (fitness per generation colored by class, fitness by class, locomotion versus
  manipulation fitness, flat versus hilly terrain).
- `python -m robogene.sim2real --entry <robogene.json> --mapping configs/sim2real_vehicle.yaml`
  rolls the trained policy out in simulation, records its wheel actions and writes the
  hardware command sequence (rescaled wheel actions, constant limb command).
