"""Goal-directed locomotion and loco-manipulation environments."""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from . import phenotype as phenotype_mod
from .phenotype import Phenotype

OBS_CLIP = 10.0
GOAL_SCALE = 10.0  # m
MAX_QVEL = 1.0e3  # joint speed treated as a numerical blow-up
MAX_GOAL_TRIES = 1000


class RoboGeneEnv(gym.Env):
    """Gymnasium environment that simulates one phenotype under lunar gravity."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, phenotype: Phenotype, cfg: dict, render_mode: str | None = None):
        super().__init__()
        self.ph = phenotype
        self.cfg = cfg
        self.task = cfg["task"]
        self.render_mode = render_mode
        sim = cfg["simulation"]
        self.frame_skip = int(sim["frame_skip"])
        self.settle_steps = int(round(sim["settle_time"] / sim["timestep"]))
        self.episode_length = int(cfg["episode"]["length"])
        self.reward_cfg = cfg["reward"]
        self.goal_cfg = cfg["goal"]
        self.cos_fall = float(np.cos(np.deg2rad(self.reward_cfg["fall_tilt_deg"])))

        self.model = phenotype_mod.build_model(phenotype, cfg)
        self.data = mujoco.MjData(self.model)
        m = self.model

        def obj_id(kind, name: str) -> int:
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise ValueError(f"model has no {kind.name} named {name!r}")
            return i

        self.act_ids = np.array([obj_id(mujoco.mjtObj.mjOBJ_ACTUATOR, a.name) for a in phenotype.actuators], int)
        self.n_act = len(self.act_ids)
        joint_ids = [obj_id(mujoco.mjtObj.mjOBJ_JOINT, a.joint) for a in phenotype.actuators]
        self.qvel_adr = np.array([m.jnt_dofadr[j] for j in joint_ids], int)
        self.vel_limit = np.array([max(a.velocity_limit, 1e-6) for a in phenotype.actuators])
        ctrl_range = m.actuator_ctrlrange[self.act_ids] if self.n_act else np.zeros((0, 2))
        self.ctrl_lo, self.ctrl_hi = ctrl_range[:, 0].copy(), ctrl_range[:, 1].copy()
        is_pos = np.array([a.control_mode == "position" for a in phenotype.actuators], bool)
        # zero action at rest: position actuators hold the range midpoint, wheels stand still
        self.rest_ctrl = np.where(is_pos, 0.5 * (self.ctrl_lo + self.ctrl_hi), 0.0)
        self.pos_qpos_adr = np.array([m.jnt_qposadr[j] for j, p in zip(joint_ids, is_pos) if p], int)
        self.pos_mid = 0.5 * (self.ctrl_lo + self.ctrl_hi)[is_pos]
        self.pos_half = np.maximum(0.5 * (self.ctrl_hi - self.ctrl_lo)[is_pos], 1e-6)

        self.base_site = obj_id(mujoco.mjtObj.mjOBJ_SITE, "base")
        root_joint = obj_id(mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_body = int(m.jnt_bodyid[root_joint])
        self.root_qpos_adr = int(m.jnt_qposadr[root_joint])
        self.root_qvel_adr = int(m.jnt_dofadr[root_joint])
        ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.ee_site = ee if ee >= 0 else None

        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self.ground_geoms = np.array([n.startswith("ground") for n in names], bool)
        flag_bodies = [int(m.jnt_bodyid[j]) for j, a in zip(joint_ids, phenotype.actuators) if a.category == "wheel"]
        if self.ee_site is not None:
            flag_bodies.append(int(m.site_bodyid[self.ee_site]))
        self.flag_bodies = np.array(flag_bodies, int)

        n_obs = len(self.pos_qpos_adr) + self.n_act + 3 + 3 + 3 + len(self.flag_bodies)
        if self.task == "loco_manip":
            n_obs += 3
        self.observation_space = gym.spaces.Box(-OBS_CLIP, OBS_CLIP, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.n_act,), np.float32)

        self.goal = np.zeros(2)
        self.target = np.zeros(3)
        self.prev_action = np.zeros(self.n_act)
        self.steps = 0
        self.goal_dist = self.target_dist = 0.0
        self.spawn_z_axis = np.array([0.0, 0.0, 1.0])
        self._renderer = None

    def action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map an action in [-1, 1] linearly onto each actuator ctrl range."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return self.ctrl_lo + 0.5 * (a + 1.0) * (self.ctrl_hi - self.ctrl_lo)

    def base_pos(self) -> np.ndarray:
        """World position of the base site."""
        return self.data.site_xpos[self.base_site].copy()

    def ee_pos(self) -> np.ndarray | None:
        """World position of the end-effector site, None without one."""
        return None if self.ee_site is None else self.data.site_xpos[self.ee_site].copy()

    def base_rot(self) -> np.ndarray:
        """Rotation matrix of the base frame in the world frame."""
        return self.data.site_xmat[self.base_site].reshape(3, 3)

    def tilt_cos(self) -> float:
        """Cosine of the angle between the root z axis now and at spawn."""
        z_now = self.data.xmat[self.root_body].reshape(3, 3)[:, 2]
        return float(np.dot(z_now, self.spawn_z_axis))

    def ground_contacts(self) -> tuple[np.ndarray, bool]:
        """Ground-contact flags of the flag bodies, and whether two robot geoms touch."""
        m, d = self.model, self.data
        flags = np.zeros(len(self.flag_bodies))
        collision = False
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = c.geom1, c.geom2
            ground1, ground2 = self.ground_geoms[g1], self.ground_geoms[g2]
            if ground1 or ground2:
                body = m.geom_bodyid[g2 if ground1 else g1]
                flags[self.flag_bodies == body] = 1.0
            elif m.geom_bodyid[g1] > 0 and m.geom_bodyid[g2] > 0:
                # MuJoCo already filters parent-child body pairs
                collision = True
        return flags, collision

    def _settle(self) -> None:
        self.data.ctrl[self.act_ids] = self.rest_ctrl
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)

    def _sample_goal(self, spawn_xy: np.ndarray) -> np.ndarray:
        g = self.goal_cfg
        goal = spawn_xy
        for _ in range(MAX_GOAL_TRIES):
            goal = np.array([self.np_random.uniform(*g["x_range"]), self.np_random.uniform(*g["y_range"])])
            if np.linalg.norm(goal - spawn_xy) >= g["min_distance"]:
                break
        return goal

    def _target_dist(self) -> float:
        ee = self.ee_pos()
        return 0.0 if ee is None else float(np.linalg.norm(ee - self.target))

    def _goal_dist(self) -> float:
        return float(np.linalg.norm(self.base_pos()[:2] - self.goal))

    def _blew_up(self, warnings_before: int) -> bool:
        d = self.data
        bad_qacc = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > warnings_before
        return bool(
            bad_qacc
            or not np.all(np.isfinite(d.qpos))
            or not np.all(np.isfinite(d.qvel))
            or np.max(np.abs(d.qvel), initial=0.0) > MAX_QVEL
        )

    def _obs(self, flags: np.ndarray) -> np.ndarray:
        d = self.data
        R = self.base_rot()
        q = d.qpos[self.pos_qpos_adr]
        qdot = d.qvel[self.qvel_adr]
        gravity_dir = R.T @ np.array([0.0, 0.0, -1.0])
        base_vel = R.T @ d.qvel[self.root_qvel_adr:self.root_qvel_adr + 3]  # free joint: world frame
        goal_vec = np.append(self.goal - self.base_pos()[:2], 0.0)
        goal_local = (R.T @ goal_vec)[:2] / GOAL_SCALE
        parts = [
            (q - self.pos_mid) / self.pos_half,
            qdot / self.vel_limit,
            gravity_dir,
            base_vel,
            goal_local,
            [np.linalg.norm(goal_vec) / GOAL_SCALE],
            flags,
        ]
        if self.task == "loco_manip":
            ee = self.ee_pos()
            rel = np.zeros(3) if ee is None else R.T @ (self.target - ee) / GOAL_SCALE
            parts.append(rel)
        obs = np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])
        obs = np.nan_to_num(obs, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP)
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _info(self, progress: float, arm_progress: float, fall: bool, collision: bool) -> dict:
        return {
            "progress": progress,
            "arm_progress": arm_progress,
            "fall": fall,
            "collision": collision,
            "base_pos": self.base_pos(),
            "ee_pos": self.ee_pos(),
            "goal": self.goal.copy(),
            "target": self.target.copy(),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        self.spawn_z_axis = d.xmat[self.root_body].reshape(3, 3)[:, 2].copy()
        spawn_xy = self.base_pos()[:2]
        self._settle()
        mujoco.mj_forward(m, d)

        if "goal" in options:
            self.goal = np.asarray(options["goal"], dtype=float)[:2].copy()
        else:
            self.goal = self._sample_goal(spawn_xy)
        if self.task == "loco_manip":
            if "target_height" in options:
                height = float(options["target_height"])
            else:
                height = float(self.np_random.uniform(*self.goal_cfg["target_height"]))
            from . import terrain

            ground = terrain.ground_height(self.cfg["terrain"], float(self.goal[0]), float(self.goal[1]))
            self.target = np.array([self.goal[0], self.goal[1], ground + height])
        else:
            self.target = np.zeros(3)

        self.prev_action = np.zeros(self.n_act)
        self.steps = 0
        self.goal_dist, self.target_dist = self._goal_dist(), self._target_dist()
        flags, collision = self.ground_contacts()
        return self._obs(flags), self._info(0.0, 0.0, False, collision)

    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, dtype=float).reshape(self.n_act), -1.0, 1.0)
        dist_before, arm_before = self.goal_dist, self.target_dist
        warnings_before = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number

        d.ctrl[self.act_ids] = self.action_to_ctrl(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(m, d)
        self.steps += 1
        truncated = self.steps >= self.episode_length

        if self._blew_up(warnings_before):
            # MuJoCo may have reset the state, so no progress is credited
            flags = np.zeros(len(self.flag_bodies))
            info = self._info(0.0, 0.0, False, False)
            info["blowup"] = True
            return self._obs(flags), 0.0, True, truncated, info

        self.goal_dist, self.target_dist = self._goal_dist(), self._target_dist()
        progress = dist_before - self.goal_dist
        arm_progress = arm_before - self.target_dist if self.task == "loco_manip" else 0.0
        fall = self.tilt_cos() < self.cos_fall
        flags, collision = self.ground_contacts()

        rc = self.reward_cfg
        smooth = float(np.sum((action - self.prev_action) ** 2))
        if self.task == "loco_manip":
            qdot = d.qvel[self.qvel_adr]
            reward = (
                rc["alpha"] * progress
                + rc["w_arm"] * arm_progress
                - rc["beta"] * smooth
                - rc["gamma_r"] * float(np.sum(qdot**2))
            )
        else:
            reward = rc["alpha"] * progress - rc["beta"] * smooth - rc["delta"] * fall - rc["zeta"] * collision
        self.prev_action = action

        terminated = bool(fall and rc.get("terminate_on_fall", False))
        info = self._info(progress, arm_progress, bool(fall), collision)
        return self._obs(flags), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._camera = mujoco.MjvCamera()
            self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._camera.trackbodyid = self.root_body
            self._camera.distance = 3.0
            self._camera.elevation = -25.0
            self._camera.azimuth = 135.0
        self._renderer.update_scene(self.data, self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
