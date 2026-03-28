"""Phenotype: the MuJoCo model of a genotype, and its validity checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import terrain
from .genotype import Genotype, arm_chains, edge_transform, forward_kinematics, morphology_class
from .schema import Library, ModuleEntry, default_library

SPAWN_GAP = 0.02
SITE_SIZE = 0.015
PENETRATION_TOL = 0.002
BLOWUP_SPEED = 100.0


@dataclass
class ActuatorInfo:
    name: str  # "a{i}"
    joint: str  # "j{i}"
    node: int
    module: str
    category: str  # "wheel" | "arm" | "hinge"
    control_mode: str  # "position" | "velocity"
    ctrl_range: tuple[float, float]
    velocity_limit: float


@dataclass
class Phenotype:
    genotype: dict  # Genotype.to_dict(), keeps the object picklable
    morphology_class: str
    robot_xml: str  # the robot <body> tree only (worldbody content)
    actuator_xml: str  # <actuator> children
    actuators: list[ActuatorInfo]  # same order as MuJoCo actuators
    end_effector: int | None  # node id whose far end carries site "ee"
    total_mass: float
    spawn_clearance: float  # m from the root origin down to the lowest point
    declared_mass: dict[int, float] = field(default_factory=dict)
    declared_inertia: dict[int, list] = field(default_factory=dict)

    @property
    def n_actuators(self) -> int:
        return len(self.actuators)

    def genotype_obj(self) -> Genotype:
        return Genotype.from_dict(self.genotype)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.10g}" for v in np.ravel(values))


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).ravel())
    return q


def _limit_scale(params: dict) -> float:
    return float(np.clip(params.get("limit_scale", 1.0), 1e-6, 1.0))


def _joint_range(mod: ModuleEntry, params: dict) -> tuple[float, float] | None:
    if mod.actuation["joint"] != "hinge" or mod.actuation.get("range") is None:
        return None
    lo, hi = mod.actuation["range"]
    s = _limit_scale(params)
    return float(lo) * s, float(hi) * s


def _lowest_extent(mod: ModuleEntry, R: np.ndarray) -> float:
    s = mod.size
    if mod.shape == "box":
        return float(np.abs(R[2]) @ np.asarray(s))
    az = abs(float(R[2, 2]))
    if mod.shape == "cylinder":
        return az * s[1] + s[0] * np.sqrt(max(0.0, 1.0 - az * az))
    return az * s[1] + s[0]


def _far_port(mod: ModuleEntry, attach_port: str) -> str:
    origin = mod.ports[attach_port].position
    return max(mod.ports, key=lambda p: np.linalg.norm(mod.ports[p].position - origin))


def _end_effector(g: Genotype, library: Library) -> tuple[int | None, float, np.ndarray | None]:
    chains = arm_chains(g, library)
    if not chains:
        return None, 0.0, None
    chain = min(chains, key=lambda c: (-len(c), c[0]))
    reach, tip = 0.0, None
    for nid in chain:
        mod = library[g.node(nid).module]
        attach = g.parent_edge(nid).child_port if nid != g.root else next(iter(mod.ports))
        tip = mod.ports[_far_port(mod, attach)].position
        reach += float(np.linalg.norm(tip - mod.ports[attach].position))
    return chain[-1], reach, tip


def expand(g: Genotype, library: Library | None = None) -> Phenotype:
    """Build the MJCF body tree and actuators of a genotype."""
    library = library or default_library()
    ee, _, ee_pos = _end_effector(g, library)
    actuators: list[ActuatorInfo] = []
    act_lines: list[str] = []
    declared_mass: dict[int, float] = {}
    declared_inertia: dict[int, list] = {}

    def body_xml(nid: int, depth: int) -> list[str]:
        node = g.node(nid)
        mod = library[node.module]
        act = mod.actuation
        ind = "  " * depth
        if nid == g.root:
            lines = [f'{ind}<body name="n{nid}" pos="0 0 0">', f'{ind}  <freejoint name="root"/>']
            lines.append(f'{ind}  <site name="base" pos="0 0 0" size="{SITE_SIZE}" rgba="0.1 0.6 1 1"/>')
        else:
            e = g.parent_edge(nid)
            R, t = edge_transform(g, e, library)
            lines = [f'{ind}<body name="n{nid}" pos="{_fmt(t)}" quat="{_fmt(_quat(R))}">']
            if mod.actuated:
                lines.append(_joint_xml(nid, mod, node.params, e.child_port, ind + "  "))
                actuators.append(_actuator_info(nid, mod, node.params))
                act_lines.append(_actuator_xml(actuators[-1], act))
        geom = mod.geometry
        lines.append(
            f'{ind}  <geom name="g{nid}" type="{mod.shape}" size="{_fmt(mod.size)}" '
            f'density="{float(geom["density"]):.10g}" rgba="{_fmt(geom.get("rgba", [0.5, 0.5, 0.5, 1]))}"/>'
        )
        if nid == ee:
            lines.append(f'{ind}  <site name="ee" pos="{_fmt(ee_pos)}" size="{SITE_SIZE}" rgba="1 0.2 0.2 1"/>')
        declared_mass[nid] = float(mod.mass())
        declared_inertia[nid] = [float(v) for v in mod.inertia()]
        for child in g.children(nid):
            lines.extend(body_xml(child, depth + 1))
        lines.append(f"{ind}</body>")
        return lines

    robot_xml = "\n".join(body_xml(g.root, 0))
    return Phenotype(
        genotype=g.to_dict(),
        morphology_class=morphology_class(g, library),
        robot_xml=robot_xml,
        actuator_xml="\n".join(act_lines),
        actuators=actuators,
        end_effector=ee,
        total_mass=float(sum(declared_mass.values())),
        spawn_clearance=_spawn_clearance(g, library),
        declared_mass=declared_mass,
        declared_inertia=declared_inertia,
    )


def _joint_xml(nid: int, mod: ModuleEntry, params: dict, child_port: str, ind: str) -> str:
    act = mod.actuation
    pos = mod.ports[child_port].position if act.get("anchor") == "port" else np.zeros(3)
    rng = _joint_range(mod, params)
    limit = f' range="{_fmt(rng)}"' if rng is not None else ""
    return (
        f'{ind}<joint name="j{nid}" type="hinge" pos="{_fmt(pos)}" axis="{_fmt(act["axis"])}"{limit} '
        f'damping="{float(act.get("damping") or 0.0):.10g}" armature="{float(act.get("armature") or 0.0):.10g}"/>'
    )


def _actuator_info(nid: int, mod: ModuleEntry, params: dict) -> ActuatorInfo:
    act = mod.actuation
    vmax = float(act.get("velocity_limit") or 0.0)
    if act["control_mode"] == "position":
        rng = _joint_range(mod, params) or (-np.pi, np.pi)
    else:
        rng = (-vmax, vmax)
    return ActuatorInfo(
        name=f"a{nid}",
        joint=f"j{nid}",
        node=nid,
        module=mod.name,
        category=mod.category,
        control_mode=act["control_mode"],
        ctrl_range=(float(rng[0]), float(rng[1])),
        velocity_limit=vmax,
    )


def _actuator_xml(info: ActuatorInfo, act: dict) -> str:
    tau = float(act["torque_limit"])
    common = f'name="{info.name}" joint="{info.joint}" ctrlrange="{_fmt(info.ctrl_range)}" forcerange="{_fmt([-tau, tau])}"'
    if info.control_mode == "position":
        return f'<position {common} kp="{float(act["kp"]):.10g}"/>'
    return f'<velocity {common} kv="{float(act["kv"]):.10g}"/>'


def _spawn_clearance(g: Genotype, library: Library) -> float:
    poses = forward_kinematics(g, library)
    lowest = min(p[2] - _lowest_extent(library[g.node(nid).module], R) for nid, (R, p) in poses.items())
    return float(-lowest)


def _footprint(g: Genotype, library: Library) -> tuple[np.ndarray, np.ndarray]:
    poses = forward_kinematics(g, library)
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for nid, (_, p) in poses.items():
        r = library[g.node(nid).module].bounding_radius()
        lo = np.minimum(lo, p[:2] - r)
        hi = np.maximum(hi, p[:2] + r)
    return lo, hi


def model_xml(ph: Phenotype, cfg: dict, spawn_xy=(0.0, 0.0)) -> str:
    """Full MJCF document: options, terrain, robot at spawn_xy and actuators."""
    sim, tcfg = cfg["simulation"], cfg["terrain"]
    lo, hi = _footprint(ph.genotype_obj(), default_library())
    x, y = float(spawn_xy[0]), float(spawn_xy[1])
    h = terrain.max_ground_height(tcfg, (x + lo[0], x + hi[0]), (y + lo[1], y + hi[1]))
    z = h + ph.spawn_clearance + SPAWN_GAP
    return f"""<mujoco model="robogene">
<compiler angle="radian" autolimits="true"/>
<option timestep="{float(sim['timestep']):.10g}" gravity="0 0 {-float(sim['gravity']):.10g}" integrator="{sim['integrator']}"/>
<default>
  <geom condim="3"/>
</default>
<asset>
{terrain.terrain_assets_xml(tcfg)}
</asset>
<worldbody>
<light name="sun" pos="0 0 10" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
{terrain.terrain_geom_xml(tcfg)}
<frame pos="{_fmt([x, y, z])}">
{ph.robot_xml}
</frame>
</worldbody>
<actuator>
{ph.actuator_xml}
</actuator>
</mujoco>
"""


def build_model(ph: Phenotype, cfg: dict, spawn_xy=(0.0, 0.0)) -> mujoco.MjModel:
    """Compile the full MJCF document of a phenotype."""
    return mujoco.MjModel.from_xml_string(model_xml(ph, cfg, spawn_xy))


def _rest_ctrl(ph: Phenotype) -> np.ndarray:
    return np.array([0.5 * sum(a.ctrl_range) if a.control_mode == "position" else 0.0 for a in ph.actuators])


def _support_metrics(points: np.ndarray, com: np.ndarray) -> tuple[float, float] | None:
    if len(points) < 3:
        return None
    centered = points - points.mean(axis=0)
    if np.linalg.svd(centered, compute_uv=False)[1] < 1e-6:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    margin = float(np.min(-(hull.equations[:, :2] @ com + hull.equations[:, 2])))
    verts = points[hull.vertices]
    width = min(float(np.max(np.abs(verts @ eq[:2] + eq[2]))) for eq in hull.equations)
    return margin, width


def check_phenotype(ph: Phenotype, cfg: dict) -> tuple[bool, list[str], dict]:
    """Phenotype checks enabled in cfg["checks"]; returns (ok, reasons, metrics)."""
    checks = cfg["checks"]
    reasons: list[str] = []
    metrics: dict[str, float] = {"mass": ph.total_mass}
    try:
        m = build_model(ph, cfg)
    except Exception as exc:
        return False, [f"model does not compile: {exc}"], metrics
    d = mujoco.MjData(m)

    bodies = {nid: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"n{nid}") for nid in ph.declared_mass}
    mass_dev = max(abs(m.body_mass[b] - ph.declared_mass[n]) / ph.declared_mass[n] for n, b in bodies.items())
    inertia_dev = 0.0
    for n, b in bodies.items():
        got, want = np.sort(m.body_inertia[b]), np.sort(ph.declared_inertia[n])
        inertia_dev = max(inertia_dev, float(np.max(np.abs(got - want) / want)))
    metrics.update(max_mass_dev=float(mass_dev), max_inertia_dev=inertia_dev)
    if checks.get("mass_inertia", True):
        tol = checks.get("mass_inertia_tol", 0.10)
        if mass_dev > tol:
            reasons.append(f"mass deviates by {mass_dev:.3f}")
        if inertia_dev > tol:
            reasons.append(f"inertia deviates by {inertia_dev:.3f}")

    root = bodies[ph.genotype["root"]]
    mujoco.mj_forward(m, d)
    z_before = d.xmat[root].reshape(3, 3)[:, 2].copy()
    d.ctrl[:] = _rest_ctrl(ph)
    steps = int(round(cfg["simulation"]["settle_time"] / m.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))) or np.max(np.abs(d.qvel)) > BLOWUP_SPEED:
        return False, reasons + ["simulation blew up while settling"], metrics

    z_after = d.xmat[root].reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(z_before @ z_after, -1.0, 1.0))))
    ground = {i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("ground")}
    contacts, penetration = [], 0.0
    for c in d.contact[: d.ncon]:
        if (c.geom1 in ground) != (c.geom2 in ground):
            contacts.append(c.pos[:2].copy())
        elif c.geom1 not in ground:
            penetration = max(penetration, -float(c.dist))
    com = d.subtree_com[root][:2]
    support = _support_metrics(np.array(contacts).reshape(-1, 2), com)
    margin, width = support if support is not None else (float("-inf"), 0.0)
    metrics.update(
        tilt_deg=tilt,
        com_margin=margin,
        support_width=width,
        n_ground_contacts=len(contacts),
        max_penetration=penetration,
    )

    if checks.get("self_penetration", True) and penetration > PENETRATION_TOL:
        reasons.append(f"self-penetration of {penetration:.4f} m")
    if checks.get("stability", False):
        if tilt > checks.get("max_tilt_deg", 5.0):
            reasons.append(f"base tilts by {tilt:.1f} deg at rest")
        if support is None:
            reasons.append("fewer than 3 non-collinear ground contacts")
        elif margin <= checks.get("com_margin", 0.05) * width:
            reasons.append(f"center of mass margin {margin:.3f} m below {checks.get('com_margin', 0.05):g} x width")

    _, reach, _ = _end_effector(ph.genotype_obj(), default_library())
    metrics["reach"] = reach
    if checks.get("reachability", False):
        if ph.end_effector is None:
            reasons.append("no end effector")
        elif reach < checks.get("reach_min", 0.3):
            reasons.append(f"arm reach {reach:.3f} m below {checks.get('reach_min', 0.3):g} m")
    return not reasons, reasons, metrics
