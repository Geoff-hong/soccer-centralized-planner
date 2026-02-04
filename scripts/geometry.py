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


def lane_clear(p_from, p_to, defenders, margin=0.7):
    defenders = np.asarray(defenders, dtype=np.float32)
    if defenders.size == 0:
        return True
    for d in defenders:
        if point_segment_dist(d, p_from, p_to) <= margin:
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
    clear = lane_clear(passer_pos, recv_pos, defenders, margin=0.7)
    d = dist(passer_pos, recv_pos)
    progress = (np.asarray(recv_pos, dtype=np.float32)[0] + 52.5) / 105.0
    score = 1.0 - 0.02 * d + 0.5 * progress
    if not clear:
        score -= 0.7
    return float(score)
