"""Boxes, answer parsing and image-token geometry (no torch needed).

Boxes are [x1, y1, x2, y2] in pixels, origin at the top-left corner, x to the right and y downward.
"""
import json
import math
import re
import zlib


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def item_seed(key, offset=0):
    """Seed of the sampled answer of one item: CRC-32 of its sampling key plus an offset."""
    return (zlib.crc32(key.encode()) + offset) % (2 ** 31)


def smart_resize(h, w, factor, min_pixels, max_pixels):
    """Resize rule of the Qwen2.5-VL / Qwen3-VL processors: sides are multiples of factor, area within [min, max]."""
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def scale_box(box, sx, sy):
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def shift_box(box, dx=0.0, dy=0.0):
    return None if box is None else [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def disagreement(base_box, view_box):
    """1 - IoU between the box of the unchanged view and the box of another view mapped back to its frame; 1 if the other view has no box."""
    if not view_box:
        return 1.0
    a, b = base_box, view_box
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 1.0 - (inter / union if union > 0 else 0.0)


_NUM = r"-?\d+(?:\.\d+)?"


def parse_box_qwen(text):
    """First box in a Qwen answer: "bbox_2d": [..] first, then any bracketed four numbers, then (x1,y1),(x2,y2).
    Returns [x1, y1, x2, y2] with x1 <= x2 and y1 <= y2, or None."""
    if not text:
        return None
    pats = [r'"bbox_2d"\s*:\s*\[\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*\]',
            r'[\[\(]\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*[\]\)]',
            r'\(\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*\)\s*,?\s*\(\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*\)']
    for p in pats:
        m = re.search(p, text)
        if m:
            x1, y1, x2, y2 = (float(g) for g in m.groups())
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None


def qwen_to_pixels(box, coord_mode, W, H):
    """Model coordinates -> pixels of the image shown. coord_mode: 'abs' (Qwen2.5-VL) or 'norm1000' (Qwen3-VL)."""
    if box is None:
        return None
    if coord_mode == "abs":
        return list(box)
    if coord_mode == "norm1000":
        return [box[0] * W / 1000, box[1] * H / 1000, box[2] * W / 1000, box[3] * H / 1000]
    raise ValueError(coord_mode)


_BRACKET_BOX = re.compile(r"\[\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_box_internvl(text):
    """First [x1, y1, x2, y2] (optionally doubly bracketed) on the 0-1000 grid; None if absent or degenerate."""
    m = _BRACKET_BOX.search(text or "")
    if not m:
        return None
    v = [float(x) for x in m.groups()]
    if v[2] <= v[0] or v[3] <= v[1]:
        return None
    return v


def internvl_to_pixels(box, W, H):
    """0-1000 box -> pixels of a W x H image, clipped to the image."""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    c = lambda v, hi: min(max(v, 0.0), float(hi))
    return [c(x1 * W / 1000.0, W), c(y1 * H / 1000.0, H), c(x2 * W / 1000.0, W), c(y2 * H / 1000.0, H)]


def parse_box_llava(text):
    """First [x1, y1, x2, y2] as four fractions; None when absent, degenerate or not on the 0-1 scale."""
    m = _BRACKET_BOX.search(text or "")
    if not m:
        return None
    v = [float(x) for x in m.groups()]
    if v[2] <= v[0] or v[3] <= v[1] or max(v) > 1.5 or min(v) < -0.5:
        return None
    return v


def square_offsets(W, H):
    """Side of the padded square and the offsets of the centred image inside it."""
    S = max(W, H)
    return S, (S - W) / 2.0, (S - H) / 2.0


def llava_to_pixels(box, W, H):
    """Fractions of the padded square -> pixels of the unpadded W x H image, clipped; None if the box lies in the padding."""
    if box is None:
        return None
    S, ox, oy = square_offsets(W, H)
    v = [box[0] * S - ox, box[1] * S - oy, box[2] * S - ox, box[3] * S - oy]
    c = lambda x, hi: min(max(x, 0.0), float(hi))
    out = [c(v[0], W), c(v[1], H), c(v[2], W), c(v[3], H)]
    return out if out[2] > out[0] and out[3] > out[1] else None


def coord_token_idx(pieces, after="bbox_2d"):
    """pieces: the generated tokens decoded one by one. Indices of the tokens that carry digits of the first box:
    after the first '[' that follows `after` (or the first '[' when `after` is empty or absent), up to the next ']'."""
    text, starts = "", []
    for p in pieces:
        starts.append(len(text))
        text += p
    k = text.find(after) if after else -1
    k = text.find("[", k if k >= 0 else 0)
    if k < 0:
        return []
    e = text.find("]", k)
    e = len(text) if e < 0 else e
    return [i for i, (s, p) in enumerate(zip(starts, pieces)) if s + len(p) > k and s < e and any(ch.isdigit() for ch in p)]


def image_spans(input_ids_row, image_token_id):
    """Contiguous runs [(start, end)] of image tokens in one row of input ids."""
    idx = [i for i, t in enumerate(input_ids_row) if t == image_token_id]
    spans = []
    if not idx:
        return spans
    start = prev = idx[0]
    for p in idx[1:]:
        if p != prev + 1:
            spans.append((start, prev + 1))
            start = p
        prev = p
    spans.append((start, prev + 1))
    return spans


def cells_in_box(rows, cols, factor, box):
    """1 for every image token (row, col) whose centre lies inside the box, else 0."""
    out = []
    for r, c in zip(rows, cols):
        cx, cy = (c + 0.5) * factor, (r + 0.5) * factor
        out.append(int(box[0] <= cx <= box[2] and box[1] <= cy <= box[3]))
    return out


def parse_point(text):
    """First point in a Qwen answer: "point_2d": [x, y] first, then <point x=".." y="..">, then a bracketed pair
    (an answer with four bracketed numbers is a box and gives no point). Returns [x, y] or None."""
    if not text:
        return None
    m = re.search(r'"point_2d"\s*:\s*\[\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*\]', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    m = re.search(r'<points?\s+x1?\s*=\s*"?(' + _NUM + r')"?\s+y1?\s*=\s*"?(' + _NUM + r')"?', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    if re.search(r'[\[\(]\s*' + _NUM + r'\s*,\s*' + _NUM + r'\s*,\s*' + _NUM + r'\s*,\s*' + _NUM + r'\s*[\]\)]', text):
        return None
    m = re.search(r'[\[\(]\s*(' + _NUM + r')\s*,\s*(' + _NUM + r')\s*[\]\)]', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    return None


def qwen_point_to_pixels(pt, coord_mode, W, H):
    if pt is None:
        return None
    if coord_mode == "abs":
        return list(pt)
    if coord_mode == "norm1000":
        return [pt[0] * W / 1000, pt[1] * H / 1000]
    raise ValueError(coord_mode)


def inside(pt, box):
    return pt is not None and box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def point_number_tokens(pieces, n=2):
    """pieces: the generated tokens decoded one by one. For each number inside the first [...] after 'point_2d' (the
    first [...] when 'point_2d' is absent), the indices of the tokens that carry its digits; None unless there are
    exactly n numbers and no token carries digits of two of them."""
    text, starts = "", []
    for p in pieces:
        starts.append(len(text))
        text += p
    k = text.find("point_2d")
    k = text.find("[", k if k >= 0 else 0)
    if k < 0:
        return None
    e = text.find("]", k)
    e = len(text) if e < 0 else e
    spans = [(m.start() + k, m.end() + k) for m in re.finditer(r"\d+(?:\.\d+)?", text[k:e])]
    if len(spans) != n:
        return None
    groups, used = [], set()
    for s0, s1 in spans:
        idx = [i for i, (s, p) in enumerate(zip(starts, pieces)) if s < s1 and s + len(p) > s0 and any(ch.isdigit() for ch in p)]
        if not idx or used & set(idx):
            return None
        used |= set(idx)
        groups.append(idx)
    return groups
