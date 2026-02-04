import numpy as np

from geometry import clip_norm, normalize


class PointSim3v3:
    def __init__(self, dt=0.1, seed=0):
        self.L = 105.0
        self.W = 68.0
        self.half_L = self.L / 2.0
        self.half_W = self.W / 2.0
        self.goal_y_half = 3.66
        self.dt = float(dt)

        self.v_max = 7.0
        self.a_max = 10.0
        self.r_player = 0.6

        self.k_friction = 1.0
        self.r_control = 1.0
        self.r_tackle = 1.25
        self.p_tackle_base = 0.2
        self.r_intercept = 1.0
        self.pass_lock_sec = 0.6
        self.pass_lock_min_sec = 0.1
        self.pass_sender_exclude_sec = 0.2
        self.pass_intercept_delay_sec = 0.2
        self.intercept_t_min = 0.2
        self.owner_protect_sec = 0.4
        self.owner_protect_steps = 0
        self.def_possess_steps = 0
        self.r_pressure = 2.5
        self.dribble_slow_near = 0.45
        self.dribble_slow_far = 0.7

        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self.episode_seed = int(seed)
        else:
            self.episode_seed = int(self.rng.integers(0, 2**32 - 1))

        # Attackers: LW/CF/RW structure (left->right attack)
        self.pos_a = self._sample_attack_structure()
        # Defenders: more to the right, left/center/right lanes
        self.pos_d = self._sample_defend_structure()
        self.vel_a = np.zeros((3, 2), dtype=np.float32)
        self.vel_d = np.zeros((3, 2), dtype=np.float32)

        owner_idx = 1  # CF starts with the ball
        self.ball_mode = "possessed"
        self.owner_team = 0
        self.owner_idx = owner_idx
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.ball_pos = self.pos_a[owner_idx].copy()
        self.pass_lock_remaining = 0
        self.pass_lock_elapsed = 0
        self.pass_intent_team = -1
        self.pass_intent_idx = -1
        self.pass_sender_team = -1
        self.pass_sender_idx = -1
        self.pass_start_pos = self.ball_pos.copy()
        self.pass_flight_steps = 0
        self.pass_sender_exclude_steps = max(1, int(round(self.pass_sender_exclude_sec / self.dt)))
        self.pass_intercept_delay_steps = max(1, int(round(self.pass_intercept_delay_sec / self.dt)))
        self.owner_protect_steps = 0
        self.def_possess_steps = 0

        self.p_tackle = float(
            np.clip(self.p_tackle_base + self.rng.uniform(-0.05, 0.05), 0.05, 0.95)
        )

        self.t = 0
        self.last_outcome = None
        self._sync_ball_to_owner()
        return self.get_obs()

    def _sample_positions(self, n, x_range, y_range):
        positions = []
        min_dist = 2.0 * self.r_player
        for _ in range(n):
            for _ in range(200):
                p = np.array(
                    [
                        self.rng.uniform(x_range[0], x_range[1]),
                        self.rng.uniform(y_range[0], y_range[1]),
                    ],
                    dtype=np.float32,
                )
                if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                    positions.append(p)
                    break
            else:
                positions.append(p)
        return np.stack(positions, axis=0)

    def _sample_with_min_dist(self, x_range, y_range, existing):
        min_dist = 2.0 * self.r_player
        for _ in range(200):
            p = np.array(
                [
                    self.rng.uniform(x_range[0], x_range[1]),
                    self.rng.uniform(y_range[0], y_range[1]),
                ],
                dtype=np.float32,
            )
            if all(np.linalg.norm(p - q) >= min_dist for q in existing):
                return p
        return p

    def _sample_attack_structure(self):
        positions = []
        # LW
        positions.append(self._sample_with_min_dist((-25.0, -8.0), (-30.0, -12.0), positions))
        # CF
        positions.append(self._sample_with_min_dist((-40.0, -25.0), (-6.0, 6.0), positions))
        # RW
        positions.append(self._sample_with_min_dist((-25.0, -8.0), (12.0, 30.0), positions))
        return np.stack(positions, axis=0)

    def _sample_defend_structure(self):
        positions = []
        positions.append(self._sample_with_min_dist((0.0, 20.0), (-26.0, -8.0), positions))
        positions.append(self._sample_with_min_dist((0.0, 20.0), (-6.0, 6.0), positions))
        positions.append(self._sample_with_min_dist((0.0, 20.0), (8.0, 26.0), positions))
        return np.stack(positions, axis=0)

    def _goal_center(self, team):
        if team == 0:
            return np.array([self.half_L, 0.0], dtype=np.float32)
        return np.array([-self.half_L, 0.0], dtype=np.float32)

    def _sync_ball_to_owner(self):
        if self.ball_mode != "possessed" or self.owner_team < 0:
            return
        if self.owner_team == 0:
            owner_pos = self.pos_a[self.owner_idx]
            owner_vel = self.vel_a[self.owner_idx]
        else:
            owner_pos = self.pos_d[self.owner_idx]
            owner_vel = self.vel_d[self.owner_idx]
        offset_dir = normalize(owner_vel)
        if np.linalg.norm(offset_dir) < 1e-6:
            offset_dir = normalize(self._goal_center(self.owner_team) - owner_pos)
        if np.linalg.norm(offset_dir) < 1e-6:
            offset_dir = np.array([1.0, 0.0], dtype=np.float32)
        if self.owner_team == 0 and owner_pos[0] > self.half_L - 1.0:
            offset_dir = np.array([1.0, 0.0], dtype=np.float32)
        if self.owner_team == 1 and owner_pos[0] < -self.half_L + 1.0:
            offset_dir = np.array([-1.0, 0.0], dtype=np.float32)
        self.ball_pos = owner_pos + offset_dir * 0.4

    def _apply_pass(self, pass_event):
        if self.ball_mode != "possessed" or self.owner_team != 0:
            return
        owner_pos = self.pos_a[self.owner_idx]
        self.pass_sender_team = self.owner_team
        self.pass_sender_idx = self.owner_idx
        self.pass_start_pos = owner_pos.copy()
        self.pass_flight_steps = 0
        target = np.asarray(pass_event.get("target_pos", owner_pos), dtype=np.float32)
        speed = float(pass_event.get("speed", 0.0))
        direction = normalize(target - owner_pos)
        if np.linalg.norm(direction) < 1e-6:
            return
        self.ball_mode = "free"
        self.owner_team = -1
        self.owner_idx = -1
        self.ball_pos = owner_pos.copy()
        self.ball_vel = direction * speed
        if pass_event.get("kind") == "pass":
            self.pass_intent_team = 0
            self.pass_intent_idx = int(pass_event.get("receiver_id", -1))
            self.pass_lock_remaining = max(1, int(round(self.pass_lock_sec / self.dt)))
            self.pass_lock_elapsed = 0
        else:
            self.pass_intent_team = -1
            self.pass_intent_idx = -1
            self.pass_lock_remaining = 0
            self.pass_lock_elapsed = 0

    def _update_players(self, v_cmd_att, v_cmd_def):
        v_cmd_att = np.asarray(v_cmd_att, dtype=np.float32)
        v_cmd_def = np.asarray(v_cmd_def, dtype=np.float32)

        v_target_a = clip_norm(v_cmd_att, self.v_max)
        v_target_d = clip_norm(v_cmd_def, self.v_max)
        # Dribble pressure: slow carrier if defenders are close
        if self.ball_mode == "possessed" and self.owner_team == 0 and self.pos_d.size > 0:
            carrier = self.owner_idx
            dists = np.linalg.norm(self.pos_d - self.pos_a[carrier], axis=1)
            min_d = float(np.min(dists))
            if min_d < self.r_pressure:
                factor = self.dribble_slow_near if min_d < self.r_tackle * 1.1 else self.dribble_slow_far
                v_target_a[carrier] = v_target_a[carrier] * factor

        dv_a = clip_norm(v_target_a - self.vel_a, self.a_max * self.dt)
        dv_d = clip_norm(v_target_d - self.vel_d, self.a_max * self.dt)

        self.vel_a = self.vel_a + dv_a
        self.vel_d = self.vel_d + dv_d

        self.pos_a = self.pos_a + self.vel_a * self.dt
        self.pos_d = self.pos_d + self.vel_d * self.dt

        self._clamp_positions()

    def _clamp_positions(self):
        for pos, vel in ((self.pos_a, self.vel_a), (self.pos_d, self.vel_d)):
            for axis, limit in enumerate([self.half_L, self.half_W]):
                low = -limit
                high = limit
                mask_low = pos[:, axis] < low
                mask_high = pos[:, axis] > high
                if mask_low.any() or mask_high.any():
                    pos[:, axis] = np.clip(pos[:, axis], low, high)
                    vel[:, axis] = np.where(mask_low | mask_high, 0.0, vel[:, axis])

    def _apply_collisions(self):
        pos = np.concatenate([self.pos_a, self.pos_d], axis=0)
        vel = np.concatenate([self.vel_a, self.vel_d], axis=0)
        n = pos.shape[0]
        min_dist = 2.0 * self.r_player
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[j] - pos[i]
                d = float(np.linalg.norm(delta))
                if d < min_dist:
                    if d < 1e-6:
                        unit = np.array([1.0, 0.0], dtype=np.float32)
                    else:
                        unit = delta / d
                    overlap = min_dist - d
                    pos[i] -= unit * overlap * 0.5
                    pos[j] += unit * overlap * 0.5
                    v1n = np.dot(vel[i], unit) * unit
                    v2n = np.dot(vel[j], unit) * unit
                    vel[i] -= v1n
                    vel[j] -= v2n
        self.pos_a = pos[:3]
        self.pos_d = pos[3:]
        self.vel_a = vel[:3]
        self.vel_d = vel[3:]
        self._clamp_positions()

    def _update_ball(self):
        if self.ball_mode == "possessed" and self.owner_team >= 0:
            self._sync_ball_to_owner()
            if self.owner_team == 0:
                self.ball_vel = self.vel_a[self.owner_idx].copy()
            else:
                self.ball_vel = self.vel_d[self.owner_idx].copy()
            return
        # free ball
        self.ball_pos = self.ball_pos + self.ball_vel * self.dt
        self.ball_vel = self.ball_vel * float(np.exp(-self.k_friction * self.dt))

    def _set_owner(self, team, idx):
        self.ball_mode = "possessed"
        self.owner_team = int(team)
        self.owner_idx = int(idx)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self._sync_ball_to_owner()
        self.pass_intent_team = -1
        self.pass_intent_idx = -1
        self.pass_lock_remaining = 0
        self.pass_lock_elapsed = 0
        self.pass_sender_team = -1
        self.pass_sender_idx = -1
        self.pass_flight_steps = 0
        self.owner_protect_steps = max(1, int(round(self.owner_protect_sec / self.dt)))

    def _check_intercept(self, prev_ball_pos):
        if self.ball_mode != "free" or np.linalg.norm(self.ball_vel) <= 1e-3:
            return False
        players = np.concatenate([self.pos_a, self.pos_d], axis=0)
        min_dist = None
        min_idx = None
        for i, p in enumerate(players):
            # Skip sender for a few frames to avoid instant self-intercept
            if self.pass_sender_team >= 0 and self.pass_flight_steps < self.pass_sender_exclude_steps:
                if self.pass_sender_team == 0 and i == self.pass_sender_idx:
                    continue
                if self.pass_sender_team == 1 and i == 3 + self.pass_sender_idx:
                    continue
            d, t = self._segment_dist_t(p, prev_ball_pos, self.ball_pos)
            if t < self.intercept_t_min:
                continue
            if min_dist is None or d < min_dist:
                min_dist = d
                min_idx = i
        if min_dist is not None and min_dist < self.r_intercept:
            if min_idx < 3:
                self._set_owner(0, min_idx)
            else:
                self._set_owner(1, min_idx - 3)
            return True
        return False

    def _segment_dist_t(self, p, a, b, eps=1e-8):
        p = np.asarray(p, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        ab = b - a
        denom = np.dot(ab, ab) + eps
        t = np.dot(p - a, ab) / denom
        t = float(np.clip(t, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(p - proj))
        return d, t

    def _check_tackle(self):
        if self.ball_mode != "possessed":
            return False
        if self.owner_protect_steps > 0:
            return False
        if self.owner_team == 0:
            owner_pos = self.pos_a[self.owner_idx]
            opp_pos = self.pos_d
            if opp_pos.size == 0:
                return False
            dists = np.linalg.norm(opp_pos - owner_pos, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < self.r_tackle:
                p = self.p_tackle + (1.0 - dists[min_idx] / self.r_tackle) * 0.4
                if self.rng.random() < min(0.95, p):
                    self._set_owner(1, min_idx)
                    return True
            return False
        if self.owner_team == 1:
            owner_pos = self.pos_d[self.owner_idx]
            opp_pos = self.pos_a
            if opp_pos.size == 0:
                return False
            dists = np.linalg.norm(opp_pos - owner_pos, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < self.r_tackle:
                p = self.p_tackle + (1.0 - dists[min_idx] / self.r_tackle) * 0.4
                if self.rng.random() < min(0.95, p):
                    self._set_owner(0, min_idx)
                    return True
            return False
        return False

    def _resolve_possession(self, prev_ball_pos):
        # 0) if recent pass: allow intercept, otherwise only intended receiver can control
        if self.ball_mode == "free" and self.pass_lock_remaining > 0:
            if self.pass_intent_team == 0 and 0 <= self.pass_intent_idx < 3:
                recv_pos = self.pos_a[self.pass_intent_idx]
                if self.pass_lock_elapsed >= max(1, int(round(self.pass_lock_min_sec / self.dt))):
                    if np.linalg.norm(recv_pos - self.ball_pos) < self.r_control * 1.2:
                        self._set_owner(0, self.pass_intent_idx)
                        return
            # allow intercept after a short delay if receiver didn't take it
            if self.pass_lock_elapsed >= self.pass_intercept_delay_steps:
                if self._check_intercept(prev_ball_pos):
                    return
            self.pass_lock_elapsed += 1
            self.pass_lock_remaining -= 1
            return

        # 1) if ball free: control by nearest within r_control
        if self.ball_mode == "free":
            players = np.concatenate([self.pos_a, self.pos_d], axis=0)
            dists = np.linalg.norm(players - self.ball_pos, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < self.r_control:
                if min_idx < 3:
                    self._set_owner(0, min_idx)
                else:
                    self._set_owner(1, min_idx - 3)
                return

        # 2) if ball free and in flight: intercept
        if self.ball_mode == "free":
            self._check_intercept(prev_ball_pos)

    def step(self, v_cmd_att, v_cmd_def, pass_event=None):
        if pass_event is not None:
            self._apply_pass(pass_event)

        self._update_players(v_cmd_att, v_cmd_def)
        # Tackle check before collision separation
        self._check_tackle()
        self._apply_collisions()

        prev_ball_pos = self.ball_pos.copy()
        self._update_ball()
        if self.ball_mode == "free" and self.pass_sender_team >= 0:
            self.pass_flight_steps += 1
        self._resolve_possession(prev_ball_pos)
        if self.ball_mode == "possessed" and self.owner_protect_steps > 0:
            self.owner_protect_steps -= 1
        if self.ball_mode == "possessed" and self.owner_team == 1:
            self.def_possess_steps += 1
        else:
            self.def_possess_steps = 0

        self.t += 1
        done, outcome = self._check_done()
        if done:
            self.last_outcome = outcome
        return done, outcome

    def _check_done(self):
        # goal
        if self.ball_pos[0] > self.half_L and abs(self.ball_pos[1]) < self.goal_y_half:
            return True, "goal"
        if self.def_possess_steps * self.dt >= 1.5:
            return True, "def_possess"
        # out of bounds
        if abs(self.ball_pos[1]) > self.half_W or abs(self.ball_pos[0]) > self.half_L:
            return True, "out"
        return False, None

    def get_obs(self):
        obs = np.zeros((7, 3), dtype=np.float32)
        # attackers
        obs[0:3, 0:2] = self._normalize_xy(self.pos_a)
        # defenders
        obs[3:6, 0:2] = self._normalize_xy(self.pos_d)
        # ball
        obs[6, 0:2] = self._normalize_xy(self.ball_pos)

        if self.ball_mode == "possessed":
            if self.owner_team == 0 and 0 <= self.owner_idx < 3:
                obs[self.owner_idx, 2] = 1.0
            if self.owner_team == 1 and 0 <= self.owner_idx < 3:
                obs[3 + self.owner_idx, 2] = 1.0
        obs[6, 2] = 0.0
        return obs

    def _normalize_xy(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        out = pos.copy()
        out[..., 0] = out[..., 0] / self.half_L
        out[..., 1] = out[..., 1] / self.half_W
        out = np.clip(out, -1.0, 1.0)
        return out
