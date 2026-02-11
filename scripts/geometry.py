import numpy as np


def dist(p, q):
    return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v)
    np.divide(v, n, out=out, where=n > eps)
    return out


def clip_norm(v, max_norm, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (n + eps))
    return v * scale


def point_segment_dist(p, a, b, eps=1e-8):
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = np.dot(ab, ab) + eps
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def point_segment_dist_raw_t(p, a, b, eps=1e-8):
    p = np.asarray(p, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = np.dot(ab, ab) + eps
    t_raw = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t_raw, 0.0, 1.0))
    proj = a + t * ab
    d = float(np.linalg.norm(p - proj))
    return d, t_raw


def lane_clear(p_from, p_to, defenders, margin=0.7):
    defenders = np.asarray(defenders, dtype=np.float32)
    if defenders.size == 0:
        return True
    for d in defenders:
        if point_segment_dist(d, p_from, p_to) <= margin:
            return False
    return True


def lane_clear_corridor(
    p_from,
    p_to,
    defenders,
    margin=1.4,
    t_min=0.1,
    t_max=0.9,
    endpoint_buffer=2.0,
):
    defenders = np.asarray(defenders, dtype=np.float32)
    if defenders.size == 0:
        return True
    p_from = np.asarray(p_from, dtype=np.float32)
    p_to = np.asarray(p_to, dtype=np.float32)
    for d in defenders:
        if np.linalg.norm(d - p_from) < endpoint_buffer:
            continue
        if np.linalg.norm(d - p_to) < endpoint_buffer:
            continue
        dist, t_raw = point_segment_dist_raw_t(d, p_from, p_to)
        if t_raw < t_min or t_raw > t_max:
            continue
        if dist <= margin:
            return False
    return True


def shot_cone_clear(carrier, goal, defenders, half_angle_deg=22.0, base_margin=1.4):
    defenders = np.asarray(defenders, dtype=np.float32)
    if defenders.size == 0:
        return True
    carrier = np.asarray(carrier, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    u = normalize(goal - carrier)
    dist_goal = float(np.linalg.norm(goal - carrier))
    if dist_goal < 1e-6:
        return False
    cos_half = float(np.cos(np.deg2rad(half_angle_deg)))
    for d in defenders:
        v = d - carrier
        dist_v = float(np.linalg.norm(v))
        if dist_v < 1e-6:
            return False
        proj = float(np.dot(v, u))
        if proj <= 0.0 or proj >= dist_goal:
            continue
        cosang = proj / (dist_v + 1e-8)
        if cosang < cos_half:
            continue
        perp_sq = max(dist_v * dist_v - proj * proj, 0.0)
        perp_dist = float(np.sqrt(perp_sq))
        margin = base_margin + 0.008 * proj
        if perp_dist <= margin:
            return False
    return True


def nearest_def_dist(p, defenders):
    defenders = np.asarray(defenders, dtype=np.float32)
    if defenders.size == 0:
        return 10.0
    d = np.linalg.norm(defenders - np.asarray(p, dtype=np.float32), axis=1)
    return float(np.min(d))


def shot_score(carrier_pos, defenders=None):
    carrier_pos = np.asarray(carrier_pos, dtype=np.float32)
    progress = (carrier_pos[0] + 52.5) / 105.0
    if defenders is None:
        d_def = 10.0
    else:
        d_def = nearest_def_dist(carrier_pos, defenders)
    d_def = float(np.clip(d_def, 0.0, 10.0))
    angle_penalty = abs(carrier_pos[1]) / 34.0
    score = 1.5 * progress + 0.2 * d_def - 0.8 * angle_penalty
    return float(score)


def pass_score(passer_pos, recv_pos, defenders):
    clear = lane_clear_corridor(
        passer_pos,
        recv_pos,
        defenders,
        margin=0.9,
        t_min=0.15,
        t_max=0.9,
        endpoint_buffer=2.0,
    )
    d = dist(passer_pos, recv_pos)
    progress = (np.asarray(recv_pos, dtype=np.float32)[0] + 52.5) / 105.0
    score = 1.0 - 0.02 * d + 0.5 * progress
    if not clear:
        score -= 0.25
    return float(score)
