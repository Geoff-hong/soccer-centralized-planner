import numpy as np

from geometry import (
    clip_norm,
    lane_clear,
    lane_clear_corridor,
    nearest_def_dist,
    normalize,
    pass_score,
    point_segment_dist,
    shot_cone_clear,
    shot_score,
)

TEMPLATE_TRIANGLE = 0
TEMPLATE_ONE_TWO = 1
TEMPLATE_THROUGH = 2
TEMPLATE_CUTBACK = 3

PASS_SHORT = 0
PASS_THROUGH = 1
PASS_CUTBACK = 2


def rotate(v, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def move_to(pos, target, speed):
    delta = target - pos
    d = np.linalg.norm(delta)
    if d < 1e-6:
        return np.zeros(2, dtype=np.float32)
    return delta / d * speed


class MixedTacticPolicy:
    def __init__(self, p_switch=0.02, eps_action=0.05, dt=0.1, v_max=7.0):
        self.p_switch = float(p_switch)
        self.eps_action = float(eps_action)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.rng = np.random.default_rng(0)
        self.reset(self.rng)

    def reset(self, rng):
        self.rng = rng
        self.template_id = TEMPLATE_TRIANGLE
        self.template_timer = 0
        self.fsm_state = "A"
        self.fsm_timer = 0
        self.step_count = 0
        self.carrier_id = None
        self.support_id = None
        self.runner_id = None
        self.one_two_origin = None
        self.one_two_support = None
        self.one_two_runner = None
        self.one_two_run_point = None
        self.one_two_start_pos = None
        self.one_two_min_wait_steps = 0
        self.one_two_min_progress = 0.45
        self.one_two_candidate_support = None
        self.one_two_candidate_run_point = None
        self.one_two_cooldown_steps = 0
        self.last_pass_passer = None
        self.last_pass_receiver = None
        self.last_pass_step = -10**9
        self.last_pass_passer_pos = None
        self.triangle_no_return_steps = max(1, int(round(1.5 / self.dt)))
        self.triangle_min_return_dx = 6.0
        self.triangle_receiver_safe_r = 2.3
        self.triangle_lane_margin = 2.5
        self.pass_receive_steps = 0
        self.pass_receiver = None
        self.pass_thr = 0.4 + self.rng.uniform(-0.05, 0.05)
        self.shoot_thr = 1.2 + self.rng.uniform(-0.05, 0.05)
        self.left_id = None
        self.right_id = None
        self.mid_id = None
        self.spacing_min = 6.0
        self.spacing_gain = 0.5
        self.wide_y = 20.0
        self.k_friction = 0.0
        self.k_friction_shot = 0.0
        self.lead_time_cap = 0.45
        self.lead_max_offset = 4.5
        self.lead_y_scale = 0.4
        self.lead_long_dist_ref = 28.0
        self.lead_x_bias = 0.8
        self.long_pass_dist = 24.0
        self.long_pass_speed_scale = 0.95
        self.overlap_steps = 0
        self.overlap_passer = None
        self.overlap_target = None

    def _register_pass(self, pass_event, pos_a):
        if pass_event is None:
            return
        if pass_event.get("kind") != "pass":
            return
        recv = int(pass_event.get("receiver_id", -1))
        if recv < 0 or recv > 2:
            return
        self.pass_receiver = recv
        self.pass_receive_steps = max(1, int(round(0.8 / self.dt)))
        if pass_event.get("pass_type") == PASS_SHORT:
            passer = int(pass_event.get("passer_id", -1))
            if 0 <= passer < 3:
                passer_pos = pos_a[passer]
                recv_pos = pos_a[recv]
                sign = 1.0 if recv_pos[1] >= passer_pos[1] else -1.0
                run_point = passer_pos + np.array([8.0, -sign * 4.0], dtype=np.float32)
                run_point = self._clamp_field(run_point)
                self.overlap_passer = passer
                self.overlap_target = run_point
                self.overlap_steps = max(1, int(round(0.8 / self.dt)))

        passer = int(pass_event.get("passer_id", -1))
        if 0 <= passer < 3:
            self.last_pass_passer = passer
            self.last_pass_receiver = recv
            self.last_pass_step = self.step_count
            self.last_pass_passer_pos = pos_a[passer].copy()

    def _maybe_noise(self, target):
        if self.rng.random() < self.eps_action:
            return target + self.rng.normal(scale=1.0, size=2).astype(np.float32)
        return target

    def _receiver_safe(self, recv_pos, pos_d, r):
        return nearest_def_dist(recv_pos, pos_d) >= r

    def _nearest_def_dist_front(self, carrier_pos, pos_d, x_back=0.5):
        if pos_d is None or pos_d.size == 0:
            return 10.0
        pos_d = np.asarray(pos_d, dtype=np.float32)
        mask = pos_d[:, 0] >= (carrier_pos[0] - x_back)
        if not np.any(mask):
            return 10.0
        d = np.linalg.norm(pos_d[mask] - carrier_pos, axis=1)
        return float(np.min(d))

    def _is_immediate_return(self, passer, recv):
        return (
            self.last_pass_passer is not None
            and passer == self.last_pass_receiver
            and recv == self.last_pass_passer
        )

    def _ensure_side_roles(self, pos_a):
        if self.left_id is not None:
            return
        order = np.argsort(pos_a[:, 1])
        self.left_id = int(order[0])
        self.mid_id = int(order[1])
        self.right_id = int(order[2])

    def _clamp_field(self, target):
        target = np.asarray(target, dtype=np.float32)
        target[0] = np.clip(target[0], -52.0, 52.0)
        target[1] = np.clip(target[1], -33.0, 33.0)
        return target

    def _should_shoot(self, carrier_pos, pos_d):
        # Only shoot in the final 35m, with a wider y window and clear lane
        if carrier_pos[0] <= 27.5 or abs(carrier_pos[1]) >= 18.0:
            return False
        if nearest_def_dist(carrier_pos, pos_d) < 2.2:
            return False
        goal = np.array([52.5, 0.0], dtype=np.float32)
        if not lane_clear(carrier_pos, goal, pos_d, margin=3.0):
            return False
        if not shot_cone_clear(carrier_pos, goal, pos_d, half_angle_deg=40.0, base_margin=2.8):
            return False
        for d in pos_d:
            if d[0] > carrier_pos[0] + 0.5:
                if point_segment_dist(d, carrier_pos, goal) < 3.0:
                    return False
        return True

    def _sample_tau(self, tau_min, tau_max):
        return float(self.rng.uniform(tau_min, tau_max))

    def _speed_for_time(self, dist, tau):
        if tau <= 1e-6:
            return None
        k = max(self.k_friction, 0.0)
        if k < 1e-4:
            v0 = dist / tau
        else:
            v0 = dist * k / (1.0 - np.exp(-k * tau))
        return float(v0)

    def _shot_speed(self, dist):
        # Sample flight time and invert speed using shot-specific friction
        if dist <= 1e-6:
            return 30.0
        tau = float(self.rng.uniform(0.6, 1.0))
        if dist > 30.0:
            tau = max(tau, 0.8)
        k = max(self.k_friction_shot, 0.0)
        if k < 1e-4:
            v0 = dist / tau
        else:
            v0 = dist * k / (1.0 - np.exp(-k * tau))
        return float(np.clip(v0, 30.0, 50.0))

    def _lead_target_tau(self, passer_pos, receiver_pos, receiver_vel, tau):
        t_lead = min(tau, self.lead_time_cap)
        offset = receiver_vel * t_lead
        offset = np.asarray(offset, dtype=np.float32)
        offset[1] *= self.lead_y_scale
        pass_dist = float(np.linalg.norm(receiver_pos - passer_pos))
        w = min(1.0, self.lead_long_dist_ref / max(pass_dist, 1e-6))
        offset *= w
        offset[0] += self.lead_x_bias * w
        off_norm = float(np.linalg.norm(offset))
        if off_norm > self.lead_max_offset:
            offset *= (self.lead_max_offset / off_norm)
        lead = receiver_pos + offset
        lead = self._clamp_field(lead)
        return lead

    def _pass_target_tau(self, passer_pos, receiver_pos, receiver_vel, preferred_target, tau):
        lead = self._lead_target_tau(passer_pos, receiver_pos, receiver_vel, tau)
        target = None
        if preferred_target is None:
            target = lead
        else:
            preferred = self._clamp_field(preferred_target)
            # If preferred target is basically receiver's current spot, lead the pass instead.
            if np.linalg.norm(preferred - receiver_pos) < 1.0:
                target = lead
            else:
                target = preferred
                # if preferred target not reachable in tau, fall back to lead
                if np.linalg.norm(target - receiver_pos) > self.v_max * tau + 1.0:
                    target = lead
        if np.linalg.norm(target - receiver_pos) > self.v_max * tau + 1.0:
            return None
        return target

    def _make_pass(
        self,
        pass_type,
        passer_pos,
        receiver_pos,
        receiver_vel,
        preferred_target,
        tau_range,
        pressured,
        forward_required=True,
    ):
        tau = self._sample_tau(tau_range[0], tau_range[1])
        target = self._pass_target_tau(passer_pos, receiver_pos, receiver_vel, preferred_target, tau)
        if target is None:
            return None
        if forward_required and (not pressured):
            if target[0] < passer_pos[0] + 2.0:
                return None

        dist = np.linalg.norm(target - passer_pos)
        speed = self._speed_for_time(dist, tau)
        if speed is None:
            return None
        if dist > self.long_pass_dist:
            speed *= self.long_pass_speed_scale
        return target, speed, tau

    def _offball_targets(self, ball_pos):
        # Maintain width/depth around the ball, but keep wingers wide
        wide_y = self.wide_y
        left_y = min(-wide_y, ball_pos[1] - 12.0)
        right_y = max(wide_y, ball_pos[1] + 12.0)
        left_target = self._clamp_field(np.array([ball_pos[0] + 6.0, left_y], dtype=np.float32))
        right_target = self._clamp_field(np.array([ball_pos[0] + 6.0, right_y], dtype=np.float32))
        mid_target = self._clamp_field(np.array([ball_pos[0] + 10.0, ball_pos[1] * 0.3], dtype=np.float32))
        return left_target, mid_target, right_target

    def _apply_spacing(self, pos_a, v_cmd):
        for i in range(3):
            for j in range(i + 1, 3):
                delta = pos_a[i] - pos_a[j]
                d = np.linalg.norm(delta)
                if d < 1e-6:
                    continue
                if d < self.spacing_min:
                    push = (delta / d) * (self.spacing_min - d) * self.spacing_gain
                    v_cmd[i] += push
                    v_cmd[j] -= push
        # Soft clip per player
        v_cmd = clip_norm(v_cmd, self.v_max)
        return v_cmd

    def _update_roles(self, carrier_idx, pos_a):
        if carrier_idx is None:
            self.carrier_id = None
            self.support_id = None
            self.runner_id = None
            return None, None, None

        others = [i for i in range(3) if i != carrier_idx]
        d0 = np.linalg.norm(pos_a[others[0]] - pos_a[carrier_idx])
        d1 = np.linalg.norm(pos_a[others[1]] - pos_a[carrier_idx])
        if d0 <= d1:
            cand_support = others[0]
            cand_runner = others[1]
        else:
            cand_support = others[1]
            cand_runner = others[0]

        if self.carrier_id == carrier_idx and self.support_id in others:
            prev_d = np.linalg.norm(pos_a[self.support_id] - pos_a[carrier_idx])
            cand_d = np.linalg.norm(pos_a[cand_support] - pos_a[carrier_idx])
            if prev_d <= cand_d + 1.0:
                support = self.support_id
                runner = others[0] if support == others[1] else others[1]
            else:
                support = cand_support
                runner = cand_runner
        else:
            support = cand_support
            runner = cand_runner

        self.carrier_id = carrier_idx
        self.support_id = support
        self.runner_id = runner
        return carrier_idx, support, runner

    def _maybe_switch_template(self):
        if self.rng.random() < self.p_switch:
            self.template_id = int(self.rng.integers(0, 4))
            self.template_timer = 0
            self.fsm_state = "A"
            self.fsm_timer = 0
            self.one_two_origin = None
            self.one_two_support = None
            self.one_two_runner = None
            self.one_two_run_point = None
            self.one_two_start_pos = None
            self.one_two_candidate_support = None
            self.one_two_candidate_run_point = None

    def _can_through(self, pos_a, vel_a, pos_d):
        carrier = self.carrier_id
        runner = self.runner_id
        if carrier is None or runner is None:
            return False
        carrier_pos = pos_a[carrier]
        runner_pos = pos_a[runner]
        runner_vel = vel_a[runner]
        t_pred = min(self.lead_time_cap, 0.30)
        runner_pred = runner_pos + runner_vel * t_pred
        runner_pred = np.asarray(runner_pred, dtype=np.float32)
        runner_pred[0] += 1.0
        runner_pred = self._clamp_field(runner_pred)
        defenders_x = pos_d[:, 0] if pos_d.size > 0 else np.array([-999.0])
        if runner_pos[0] <= defenders_x.max() - 1.5:
            return False
        return lane_clear(carrier_pos, runner_pred, pos_d)

    def _plan_one_two(self, pos_a, pos_d):
        carrier = self.carrier_id
        if carrier is None:
            return False
        carrier_pos = pos_a[carrier]
        pressed = nearest_def_dist(carrier_pos, pos_d) < 4.2

        best_score = None
        best_support = None
        best_run_point = None

        for support in [i for i in range(3) if i != carrier]:
            support_pos = pos_a[support]
            dist_cs = np.linalg.norm(support_pos - carrier_pos)
            if dist_cs < 6.0 or dist_cs > 12.0:
                continue
            if not lane_clear(carrier_pos, support_pos, pos_d):
                continue
            sign = 1.0 if support_pos[1] >= 0.0 else -1.0
            run_point = support_pos + np.array(
                [self.rng.uniform(6.0, 9.0), -sign * self.rng.uniform(3.0, 4.0)],
                dtype=np.float32,
            )
            run_point = self._clamp_field(run_point)
            if run_point[0] < carrier_pos[0] + 4.0:
                continue
            score = (run_point[0] - carrier_pos[0]) + 0.05 * abs(support_pos[1])
            if best_score is None or score > best_score:
                best_score = score
                best_support = support
                best_run_point = run_point

        if best_support is None:
            return False
        if not pressed and best_run_point[0] < carrier_pos[0] + 6.0:
            return False

        self.one_two_candidate_support = best_support
        self.one_two_candidate_run_point = best_run_point
        return True

    def _can_cutback(self, pos_a):
        carrier = self.carrier_id
        if carrier is None:
            return False
        carrier_pos = pos_a[carrier]
        return carrier_pos[0] > 35.0 and abs(carrier_pos[1]) > 18.0

    def _select_template(self, pos_a, vel_a, pos_d):
        # Prioritize through, then wing one-two, then cutback, else triangle
        if self._can_through(pos_a, vel_a, pos_d):
            return TEMPLATE_THROUGH
        self.one_two_candidate_support = None
        self.one_two_candidate_run_point = None
        if self.one_two_cooldown_steps <= 0 and self._plan_one_two(pos_a, pos_d):
            return TEMPLATE_ONE_TWO
        if self._can_cutback(pos_a):
            return TEMPLATE_CUTBACK
        return TEMPLATE_TRIANGLE

    def act(self, env):
        pos_a = env.pos_a
        vel_a = env.vel_a
        pos_d = env.pos_d
        ball_pos = env.ball_pos
        ball_mode = env.ball_mode
        owner_team = env.owner_team
        owner_idx = env.owner_idx
        self.k_friction = float(getattr(env, "k_friction", 0.0))
        self.k_friction_shot = float(getattr(env, "k_friction_shot", self.k_friction))

        v_cmd = np.zeros((3, 2), dtype=np.float32)
        pass_event = None

        self._ensure_side_roles(pos_a)
        self.step_count += 1
        if self.one_two_cooldown_steps > 0:
            self.one_two_cooldown_steps -= 1

        if self.pass_receive_steps > 0:
            if ball_mode == "free":
                recv = self.pass_receiver
                if recv is not None and 0 <= recv < 3:
                    left_target, mid_target, right_target = self._offball_targets(ball_pos)
                    v_cmd = np.zeros((3, 2), dtype=np.float32)
                    v_cmd[recv] = move_to(pos_a[recv], ball_pos, self.v_max)
                    for i in range(3):
                        if i == recv:
                            continue
                        if i == self.left_id:
                            v_cmd[i] = move_to(pos_a[i], left_target, self.v_max * 0.7)
                        elif i == self.right_id:
                            v_cmd[i] = move_to(pos_a[i], right_target, self.v_max * 0.7)
                        else:
                            v_cmd[i] = move_to(pos_a[i], mid_target, self.v_max * 0.7)
                    self.pass_receive_steps -= 1
                    v_cmd = self._apply_spacing(pos_a, v_cmd)
                    return v_cmd, None
                self.pass_receive_steps = 0
                self.pass_receiver = None
            else:
                if owner_team == 0 and owner_idx == self.pass_receiver:
                    self.pass_receive_steps = 0
                    self.pass_receiver = None
                elif owner_team != 0 or owner_idx != self.pass_receiver:
                    self.pass_receive_steps = 0
                    self.pass_receiver = None

        if ball_mode == "free" or owner_team != 0:
            self.last_pass_passer = None
            self.last_pass_receiver = None
            self.last_pass_passer_pos = None
            # one chaser, others hold width/depth
            dists = np.linalg.norm(pos_a - ball_pos, axis=1)
            chaser = int(np.argmin(dists))
            left_target, mid_target, right_target = self._offball_targets(ball_pos)
            for i in range(3):
                if i == chaser:
                    v_cmd[i] = move_to(pos_a[i], ball_pos, self.v_max)
                elif i == self.left_id:
                    v_cmd[i] = move_to(pos_a[i], left_target, self.v_max * 0.7)
                elif i == self.right_id:
                    v_cmd[i] = move_to(pos_a[i], right_target, self.v_max * 0.7)
                else:
                    v_cmd[i] = move_to(pos_a[i], mid_target, self.v_max * 0.7)
            self.template_id = TEMPLATE_TRIANGLE
            self.fsm_state = "A"
            self.fsm_timer = 0
            self.one_two_origin = None
            self.one_two_support = None
            self.one_two_runner = None
            self.one_two_run_point = None
            self.one_two_start_pos = None
            self.one_two_candidate_support = None
            self.one_two_candidate_run_point = None
            v_cmd = self._apply_spacing(pos_a, v_cmd)
            if self.overlap_steps > 0 and self.overlap_passer is not None:
                passer = self.overlap_passer
                if 0 <= passer < 3 and self.overlap_target is not None:
                    v_cmd[passer] = move_to(pos_a[passer], self.overlap_target, self.v_max * 0.9)
                self.overlap_steps -= 1
            return v_cmd, None

        if self.template_id == TEMPLATE_ONE_TWO and self.fsm_state == "B" and self.one_two_origin is not None:
            carrier_idx = self.one_two_origin
            carrier = carrier_idx
            support = self.one_two_support
            runner = self.one_two_runner
            prev_carrier = carrier_idx
            self.carrier_id = carrier
            self.support_id = support
            self.runner_id = runner
        else:
            carrier_idx = owner_idx
            prev_carrier = self.carrier_id
            carrier, support, runner = self._update_roles(carrier_idx, pos_a)

        if carrier is None:
            return v_cmd, None

        if prev_carrier != carrier_idx:
            self.template_id = TEMPLATE_TRIANGLE
            self.template_timer = 0
            self.fsm_state = "A"
            self.fsm_timer = 0
            self.one_two_origin = None
            self.one_two_support = None
            self.one_two_runner = None
            self.one_two_run_point = None
            self.one_two_start_pos = None
            self.one_two_candidate_support = None
            self.one_two_candidate_run_point = None
        else:
            if self.template_id != TEMPLATE_ONE_TWO:
                self.template_id = self._select_template(pos_a, vel_a, pos_d)
                if self.template_id != TEMPLATE_ONE_TWO:
                    self.one_two_candidate_support = None
                    self.one_two_candidate_run_point = None

        self.template_timer += 1

        if self.template_id == TEMPLATE_ONE_TWO:
            v_cmd, pass_event = self._act_one_two(pos_a, vel_a, pos_d, owner_team, owner_idx)
        elif self.template_id == TEMPLATE_THROUGH:
            v_cmd, pass_event = self._act_through(pos_a, vel_a, pos_d)
        elif self.template_id == TEMPLATE_CUTBACK:
            v_cmd, pass_event = self._act_cutback(pos_a, vel_a, pos_d)
        else:
            v_cmd, pass_event = self._act_triangle(pos_a, vel_a, pos_d)

        v_cmd = self._apply_spacing(pos_a, v_cmd)
        if self.overlap_steps > 0 and self.overlap_passer is not None:
            passer = self.overlap_passer
            if 0 <= passer < 3 and self.overlap_target is not None:
                v_cmd[passer] = move_to(pos_a[passer], self.overlap_target, self.v_max * 0.9)
            self.overlap_steps -= 1
        return v_cmd, pass_event

    def _triangle_targets(self, pos_a, pos_d):
        carrier = self.carrier_id
        support = self.support_id
        runner = self.runner_id
        goal = np.array([52.5, 0.0], dtype=np.float32)
        carrier_pos = pos_a[carrier]
        dir_goal = normalize(goal - carrier_pos)
        if np.linalg.norm(dir_goal) < 1e-6:
            dir_goal = np.array([1.0, 0.0], dtype=np.float32)
        perp = np.array([-dir_goal[1], dir_goal[0]], dtype=np.float32)
        sign = 1.0 if pos_a[support][1] >= carrier_pos[1] else -1.0
        theta = np.deg2rad(35.0 * sign)
        support_dir = rotate(dir_goal, theta)
        target_support = carrier_pos + support_dir * 8.0 + perp * (5.0 * sign)

        sign_r = 1.0 if pos_a[runner][1] >= carrier_pos[1] else -1.0
        if abs(sign_r) < 0.5:
            sign_r = 1.0
        target_runner = carrier_pos + np.array(
            [self.rng.uniform(6.0, 10.0), sign_r * self.rng.uniform(14.0, 20.0)],
            dtype=np.float32,
        )

        # Keep wide players wide
        if support == self.left_id:
            target_support[1] = min(target_support[1], -self.wide_y)
        elif support == self.right_id:
            target_support[1] = max(target_support[1], self.wide_y)
        if runner == self.left_id:
            target_runner[1] = min(target_runner[1], -self.wide_y)
        elif runner == self.right_id:
            target_runner[1] = max(target_runner[1], self.wide_y)
        return target_support, target_runner

    def _act_triangle(self, pos_a, vel_a, pos_d):
        carrier = self.carrier_id
        support = self.support_id
        runner = self.runner_id
        carrier_pos = pos_a[carrier]
        goal = np.array([52.5, 0.0], dtype=np.float32)
        def_dist = nearest_def_dist(carrier_pos, pos_d)
        shoot = self._should_shoot(carrier_pos, pos_d)

        # --- TRIANGLE: if carrier is already the most advanced player on x and has a clear dribble lane, forbid passing ---
        eps_front = 0.1
        max_x_pitch = float(np.max(pos_a[:, 0]))
        if pos_d is not None and pos_d.size > 0:
            max_x_pitch = max(max_x_pitch, float(np.max(pos_d[:, 0])))
        is_frontmost_pitch = carrier_pos[0] >= max_x_pitch - eps_front
        disable_triangle_pass = is_frontmost_pitch

        passer = carrier
        passer_pos = carrier_pos

        def allow_recv(recv_id, recv_pos):
            if not self._is_immediate_return(passer, recv_id):
                return True
            age_ok = (self.step_count - self.last_pass_step) >= self.triangle_no_return_steps
            pos_ok = self.last_pass_passer_pos is not None
            dx = (
                pos_a[self.last_pass_passer][0] - self.last_pass_passer_pos[0]
                if pos_ok
                else -1e9
            )
            dx_ok = dx >= self.triangle_min_return_dx
            lane_ok = lane_clear_corridor(
                passer_pos,
                recv_pos,
                pos_d,
                margin=1.2,
                t_min=0.15,
                t_max=0.85,
                endpoint_buffer=2.0,
            )
            recv_ok = self._receiver_safe(recv_pos, pos_d, self.triangle_receiver_safe_r)

            if not age_ok:
                return False
            if not pos_ok:
                return False
            if not dx_ok:
                return False
            if not lane_ok:
                return False
            if not recv_ok:
                return False
            return True

        pass_event = None
        if (not shoot) and (not disable_triangle_pass):
            dist_s = np.linalg.norm(pos_a[support] - carrier_pos)
            dist_r = np.linalg.norm(pos_a[runner] - carrier_pos)
            s_allowed = dist_s > 3.0 and allow_recv(support, pos_a[support])
            r_allowed = dist_r > 3.0 and allow_recv(runner, pos_a[runner])
            s_safe = s_allowed and self._receiver_safe(pos_a[support], pos_d, self.triangle_receiver_safe_r)
            r_safe = r_allowed and self._receiver_safe(pos_a[runner], pos_d, self.triangle_receiver_safe_r)

            def_dist_front = self._nearest_def_dist_front(carrier_pos, pos_d, x_back=0.5)
            pressured = def_dist_front < 4.8
            very_safe = def_dist_front > 6.0

            # Under pressure: force a pass to any safe receiver (no best_score threshold).
            if pressured:
                candidates = []
                if s_safe:
                    candidates.append(support)
                if r_safe:
                    candidates.append(runner)
                if candidates:
                    # pick by pass_score among safe
                    best_score = -1e9
                    recv = candidates[0]
                    for cand in candidates:
                        score = pass_score(carrier_pos, pos_a[cand], pos_d)
                        if score > best_score:
                            best_score = score
                            recv = cand
                    recv_pos = pos_a[recv]
                    recv_vel = vel_a[recv]
                    is_return = self._is_immediate_return(passer, recv)
                    tau_range = (0.7, 0.95)
                    pass_pressured = True
                    forward_required = False
                    res = self._make_pass(
                        PASS_SHORT,
                        carrier_pos,
                        recv_pos,
                        recv_vel,
                        recv_pos,
                        tau_range,
                        pass_pressured,
                        forward_required=forward_required,
                    )
                    if res is not None:
                        target, speed, _tau = res
                        pass_event = {
                            "kind": "pass",
                            "passer_id": carrier,
                            "receiver_id": recv,
                            "pass_type": PASS_SHORT,
                            "target_pos": target,
                            "speed": speed,
                        }
            else:
                # Not pressured: only pass if clearly better.
                s_score = pass_score(carrier_pos, pos_a[support], pos_d) if s_safe else -1e9
                r_score = pass_score(carrier_pos, pos_a[runner], pos_d) if r_safe else -1e9
                best_score = s_score
                recv = support
                if r_score > best_score:
                    best_score = r_score
                    recv = runner
                if def_dist < 3.2:
                    thr = self.pass_thr - 0.25
                elif def_dist < 4.2:
                    thr = self.pass_thr - 0.1
                else:
                    thr = self.pass_thr
                force_pass = def_dist < 2.0
                if best_score > thr and (best_score > self.pass_thr + 0.2 or force_pass):
                    recv_pos = pos_a[recv]
                    recv_vel = vel_a[recv]
                    tau_range = (0.7, 0.95)
                    pass_pressured = False
                    forward_required = very_safe
                    res = self._make_pass(
                        PASS_SHORT,
                        carrier_pos,
                        recv_pos,
                        recv_vel,
                        recv_pos,
                        tau_range,
                        pass_pressured,
                        forward_required=forward_required,
                    )
                    if res is not None:
                        target, speed, _tau = res
                        pass_event = {
                            "kind": "pass",
                            "passer_id": carrier,
                            "receiver_id": recv,
                            "pass_type": PASS_SHORT,
                            "target_pos": target,
                            "speed": speed,
                        }

        if shoot:
            shot_speed = self._shot_speed(np.linalg.norm(goal - carrier_pos))
            pass_event = {
                "kind": "shot",
                "passer_id": carrier,
                "receiver_id": carrier,
                "pass_type": PASS_SHORT,
                "target_pos": np.array([52.5, 0.0], dtype=np.float32),
                "speed": shot_speed,
            }

        target_support, target_runner = self._triangle_targets(pos_a, pos_d)
        target_support = self._maybe_noise(target_support)
        target_runner = self._maybe_noise(target_runner)

        v_cmd = np.zeros((3, 2), dtype=np.float32)
        carrier_speed = self.v_max * (0.75 if def_dist < 3.0 else 0.9)
        v_cmd[carrier] = move_to(carrier_pos, carrier_pos + normalize(goal - carrier_pos) * 5.0, carrier_speed)
        v_cmd[support] = move_to(pos_a[support], target_support, self.v_max * 0.7)
        v_cmd[runner] = move_to(pos_a[runner], target_runner, self.v_max)
        if pass_event is not None and pass_event.get("kind") == "pass":
            self._register_pass(pass_event, pos_a)
        return v_cmd, pass_event

    def _act_one_two(self, pos_a, vel_a, pos_d, owner_team, owner_idx):
        carrier = self.carrier_id
        if carrier is None:
            return np.zeros((3, 2), dtype=np.float32), None

        v_cmd = np.zeros((3, 2), dtype=np.float32)
        pass_event = None

        self.fsm_timer += 1

        # Determine support/runner/run_point
        if self.fsm_state == "B" and self.one_two_origin is not None:
            carrier = self.one_two_origin
            support = self.one_two_support
            runner = self.one_two_runner
            run_point = self.one_two_run_point
        else:
            support = self.one_two_candidate_support
            if support is None or support == carrier:
                others = [i for i in range(3) if i != carrier]
                d0 = np.linalg.norm(pos_a[others[0]] - pos_a[carrier])
                d1 = np.linalg.norm(pos_a[others[1]] - pos_a[carrier])
                support = others[0] if d0 <= d1 else others[1]
            runner = [i for i in range(3) if i not in (carrier, support)][0]
            if self.one_two_candidate_run_point is not None:
                run_point = self.one_two_candidate_run_point
            else:
                sign = 1.0 if pos_a[support][1] >= 0.0 else -1.0
                run_point = pos_a[support] + np.array(
                    [self.rng.uniform(6.0, 9.0), -sign * self.rng.uniform(3.0, 4.0)],
                    dtype=np.float32,
                )
                run_point = self._clamp_field(run_point)

        carrier_pos = pos_a[carrier]
        support_pos = pos_a[support]
        runner_pos = pos_a[runner]
        goal = np.array([52.5, 0.0], dtype=np.float32)

        dist_cs = np.linalg.norm(support_pos - carrier_pos)
        can_setup = 6.0 <= dist_cs <= 12.0 and lane_clear(carrier_pos, support_pos, pos_d)
        pressed = nearest_def_dist(carrier_pos, pos_d) < 4.2
        if run_point[0] < carrier_pos[0] + 4.0:
            can_setup = False
        if not pressed and run_point[0] < carrier_pos[0] + 6.0:
            can_setup = False

        if self.fsm_state == "A":
            dir_goal = normalize(goal - carrier_pos)
            target_support = support_pos
            target_runner = np.array([carrier_pos[0] + 12.0, runner_pos[1]], dtype=np.float32)

            v_cmd[carrier] = move_to(carrier_pos, carrier_pos + dir_goal * 4.0, self.v_max * 0.9)
            v_cmd[support] = move_to(support_pos, self._maybe_noise(target_support), self.v_max * 0.7)
            v_cmd[runner] = move_to(runner_pos, self._maybe_noise(target_runner), self.v_max)

            if can_setup:
                res = self._make_pass(
                    PASS_SHORT,
                    carrier_pos,
                    support_pos,
                    vel_a[support],
                    support_pos,
                    (0.37, 0.53),
                    pressured=True,
                    forward_required=True,
                )
                if res is not None:
                    target, speed, _tau = res
                    pass_event = {
                        "kind": "pass",
                        "passer_id": carrier,
                        "receiver_id": support,
                        "pass_type": PASS_SHORT,
                        "target_pos": target,
                        "speed": speed,
                    }
                    self.fsm_state = "B"
                    self.fsm_timer = 0
                    self.one_two_origin = carrier
                    self.one_two_support = support
                    self.one_two_runner = runner
                    self.one_two_run_point = run_point
                    self.one_two_start_pos = carrier_pos.copy()
                    self.one_two_min_wait_steps = max(1, int(round(0.9 / self.dt)))
                    # Start run toward run_point immediately after the pass
                    v_cmd[carrier] = move_to(carrier_pos, run_point, self.v_max)

        elif self.fsm_state == "B":
            run_point = run_point if run_point is not None else carrier_pos + normalize(goal - carrier_pos) * 8.0
            v_cmd[carrier] = move_to(carrier_pos, run_point, self.v_max)
            v_cmd[runner] = move_to(
                runner_pos,
                np.array([carrier_pos[0] + 12.0, runner_pos[1]], dtype=np.float32),
                self.v_max,
            )

            if owner_team == 0 and owner_idx == support:
                start_pos = self.one_two_start_pos if self.one_two_start_pos is not None else carrier_pos
                total = np.linalg.norm(run_point - start_pos)
                remaining = np.linalg.norm(carrier_pos - run_point)
                progress = 1.0 if total < 1e-6 else 1.0 - remaining / total
                dx = carrier_pos[0] - start_pos[0]



                ready = self.fsm_timer >= self.one_two_min_wait_steps and (
                    (progress >= self.one_two_min_progress or remaining <= 5.0) and dx >= 4.0
                )
                # if not ready:
                #     print(f"self.fsm_timer >= self.one_two_min_wait_steps: {self.fsm_timer >= self.one_two_min_wait_steps}")
                #     print(f"progress >= self.one_two_min_progress: {progress >= self.one_two_min_progress}")
                #     print(f"progress: {progress}")
                #     print(f"remaining: {remaining}")
                #     print(f"remaining <= 5.0: {remaining <= 5.0}")
                #     print(f"dx >= 4.0: {dx >= 4.0}")

                if ready and lane_clear(support_pos, run_point, pos_d):
                    res = self._make_pass(
                        PASS_SHORT,
                        support_pos,
                        carrier_pos,
                        vel_a[carrier],
                        run_point,
                        (0.37, 0.53),
                        pressured=True,
                        forward_required=True,
                    )
                    if res is not None:
                        target, speed, _tau = res
                        pass_event = {
                            "kind": "pass",
                            "passer_id": support,
                            "receiver_id": carrier,
                            "pass_type": PASS_SHORT,
                            "target_pos": target,
                            "speed": speed,
                        }
                        self.template_id = TEMPLATE_TRIANGLE
                        self.fsm_state = "A"
                        self.fsm_timer = 0
                        self.one_two_origin = None
                        self.one_two_support = None
                        self.one_two_runner = None
                        self.one_two_run_point = None
                        self.one_two_start_pos = None
                        self.one_two_candidate_support = None
                        self.one_two_candidate_run_point = None
                        self.one_two_cooldown_steps = max(1, int(round(0.4 / self.dt)))
                elif ready and lane_clear(support_pos, runner_pos, pos_d):
                    res = self._make_pass(
                        PASS_THROUGH,
                        support_pos,
                        runner_pos,
                        vel_a[runner],
                        runner_pos,
                        (0.6, 0.9),
                        pressured=True,
                        forward_required=True,
                    )
                    if res is not None:
                        target, speed, _tau = res
                        pass_event = {
                            "kind": "pass",
                            "passer_id": support,
                            "receiver_id": runner,
                            "pass_type": PASS_THROUGH,
                            "target_pos": target,
                            "speed": speed,
                        }
                        self.template_id = TEMPLATE_TRIANGLE
                        self.fsm_state = "A"
                        self.fsm_timer = 0
                        self.one_two_origin = None
                        self.one_two_support = None
                        self.one_two_runner = None
                        self.one_two_run_point = None
                        self.one_two_start_pos = None
                        self.one_two_candidate_support = None
                        self.one_two_candidate_run_point = None
                        self.one_two_cooldown_steps = max(1, int(round(0.4 / self.dt)))
                else:
                    hold_target = self._clamp_field(support_pos + np.array([1.5, 0.0], dtype=np.float32))
                    v_cmd[support] = move_to(support_pos, hold_target, self.v_max * 0.4)
            else:
                v_cmd[support] = move_to(support_pos, support_pos, 0.0)

            if self.fsm_timer > 14:
                self.template_id = TEMPLATE_TRIANGLE
                self.fsm_state = "A"
                self.fsm_timer = 0
                self.one_two_origin = None
                self.one_two_support = None
                self.one_two_runner = None
                self.one_two_run_point = None
                self.one_two_start_pos = None
                self.one_two_candidate_support = None
                self.one_two_candidate_run_point = None
                self.one_two_cooldown_steps = max(1, int(round(0.4 / self.dt)))

        if self.fsm_state == "A" and self.fsm_timer > 15:
            self.template_id = TEMPLATE_TRIANGLE
            self.fsm_state = "A"
            self.fsm_timer = 0
            self.one_two_origin = None
            self.one_two_support = None
            self.one_two_runner = None
            self.one_two_run_point = None
            self.one_two_start_pos = None
            self.one_two_candidate_support = None
            self.one_two_candidate_run_point = None

        if pass_event is not None and pass_event.get("kind") == "pass":
            self._register_pass(pass_event, pos_a)
        return v_cmd, pass_event

    def _act_through(self, pos_a, vel_a, pos_d):
        carrier = self.carrier_id
        runner = self.runner_id
        carrier_pos = pos_a[carrier]
        runner_pos = pos_a[runner]
        runner_vel = vel_a[runner]

        t_pred = min(self.lead_time_cap, 0.30)
        runner_pred = runner_pos + runner_vel * t_pred
        runner_pred = np.asarray(runner_pred, dtype=np.float32)
        runner_pred[0] += 1.0
        runner_pred = self._clamp_field(runner_pred)
        defenders_x = pos_d[:, 0] if pos_d.size > 0 else np.array([-999.0])
        cond = runner_pos[0] > defenders_x.max() - 2.0 and lane_clear(carrier_pos, runner_pred, pos_d)

        pass_event = None
        if cond:
            res = self._make_pass(
                PASS_THROUGH,
                carrier_pos,
                runner_pos,
                runner_vel,
                runner_pred,
                (0.6, 0.9),
                pressured=False,
                forward_required=True,
            )
            if res is not None:
                target, speed, _tau = res
                pass_event = {
                    "kind": "pass",
                    "passer_id": carrier,
                    "receiver_id": runner,
                    "pass_type": PASS_THROUGH,
                    "target_pos": target,
                    "speed": speed,
                }

        # fallback to triangle movement
        v_cmd, tri_pass = self._act_triangle(pos_a, vel_a, pos_d)
        if pass_event is None:
            pass_event = tri_pass
        if pass_event is not None and pass_event.get("kind") == "pass":
            self._register_pass(pass_event, pos_a)
        return v_cmd, pass_event

    def _act_cutback(self, pos_a, vel_a, pos_d):
        carrier = self.carrier_id
        support = self.support_id
        runner = self.runner_id
        carrier_pos = pos_a[carrier]
        goal = np.array([52.5, 0.0], dtype=np.float32)

        on_wing = carrier_pos[0] > 35.0 and abs(carrier_pos[1]) > 15.0
        if not on_wing:
            self.template_id = TEMPLATE_TRIANGLE
            self.one_two_origin = None
            self.one_two_support = None
            self.one_two_runner = None
            return self._act_triangle(pos_a, vel_a, pos_d)

        target_runner = np.array([45.0, 0.0], dtype=np.float32)
        target_support = np.array([38.0, -np.sign(carrier_pos[1]) * 5.0], dtype=np.float32)
        carrier_target = np.array([min(carrier_pos[0] + 2.0, 52.0), carrier_pos[1]], dtype=np.float32)

        v_cmd = np.zeros((3, 2), dtype=np.float32)
        v_cmd[carrier] = move_to(carrier_pos, carrier_target, self.v_max * 0.9)
        v_cmd[support] = move_to(pos_a[support], self._maybe_noise(target_support), self.v_max * 0.7)
        v_cmd[runner] = move_to(pos_a[runner], self._maybe_noise(target_runner), self.v_max)

        pass_event = None
        self.fsm_timer += 1
        if self.fsm_timer > 5 or nearest_def_dist(carrier_pos, pos_d) < 2.0:
            s_score = pass_score(carrier_pos, pos_a[support], pos_d)
            r_score = pass_score(carrier_pos, pos_a[runner], pos_d)
            recv = support if s_score >= r_score else runner
            res = self._make_pass(
                PASS_CUTBACK,
                carrier_pos,
                pos_a[recv],
                vel_a[recv],
                pos_a[recv],
                (0.5, 0.7),
                pressured=True,
                forward_required=True,
            )
            if res is not None:
                target, speed, _tau = res
                pass_event = {
                    "kind": "pass",
                    "passer_id": carrier,
                    "receiver_id": recv,
                    "pass_type": PASS_CUTBACK,
                    "target_pos": target,
                    "speed": speed,
                }
                self.template_id = TEMPLATE_TRIANGLE
                self.fsm_state = "A"
                self.fsm_timer = 0
                self.one_two_origin = None
                self.one_two_support = None
                self.one_two_runner = None

        if self._should_shoot(carrier_pos, pos_d):
            shot_speed = self._shot_speed(np.linalg.norm(goal - carrier_pos))
            pass_event = {
                "kind": "shot",
                "passer_id": carrier,
                "receiver_id": carrier,
                "pass_type": PASS_CUTBACK,
                "target_pos": goal,
                "speed": shot_speed,
            }
        if pass_event is not None and pass_event.get("kind") == "pass":
            self._register_pass(pass_event, pos_a)
        return v_cmd, pass_event


class DefenderPolicy:
    def __init__(self, v_max=7.0):
        self.v_max = float(v_max)
        self.rng = np.random.default_rng(1)
        self.pressure = 1.0

    def reset(self, rng):
        self.rng = rng
        self.pressure = 1.0 + self.rng.uniform(-0.1, 0.1)

    def act(self, env):
        pos_a = env.pos_a
        pos_d = env.pos_d
        ball_pos = env.ball_pos

        if env.ball_mode == "free":
            chase_target = ball_pos
        elif env.owner_team == 0:
            chase_target = pos_a[env.owner_idx]
        else:
            chase_target = ball_pos

        dists = np.linalg.norm(pos_d - chase_target, axis=1)
        chaser = int(np.argmin(dists))

        # runner = most advanced attacker
        runner_idx = int(np.argmax(pos_a[:, 0]))

        remaining = [i for i in range(3) if i != chaser]
        marker = remaining[0]
        protector = remaining[1]

        runner_pos = pos_a[runner_idx]
        # Shadow runner on the line between runner and ball (cut passing lane)
        dir_rb = normalize(ball_pos - runner_pos)
        target_marker = runner_pos + dir_rb * 3.0
        # Protector holds a central blocking position based on ball x
        protect_x = 35.0 if ball_pos[0] > 0 else 25.0
        target_protect = np.array([protect_x, ball_pos[1] * 0.3], dtype=np.float32)

        v_cmd = np.zeros((3, 2), dtype=np.float32)
        v_cmd[chaser] = move_to(pos_d[chaser], chase_target, self.v_max * self.pressure)
        v_cmd[marker] = move_to(pos_d[marker], target_marker, self.v_max * 0.8 * self.pressure)
        v_cmd[protector] = move_to(pos_d[protector], target_protect, self.v_max * 0.65)
        return v_cmd
