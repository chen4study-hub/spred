"""Item files for the model runs: one JSON line per item of a list in data/lists/, with a local copy of its image.

The RefCOCO-family images are taken from the Hugging Face copies of the datasets at the revisions given in
data/README.md, decoded to RGB and saved as JPEG (quality 95), which is how the images of every reported run were
made. Items are matched to dataset rows by COCO image id and expression; the annotated box is read from the row
(COCO [x, y, w, h] converted to [x1, y1, x2, y2]).

  python -m spred.items --list data/lists/training.csv --dataset lmms-lab-encoder/RefCOCO --revision a5dff0b31947 \
      --split val --image-dir work/images/training --out work/items/training.jsonl

Each output line holds: id (item id of the list), image_id, img (local image file), ref (expression), box, W, H and
seed_key, the key from which the seed of a sampled answer is derived ("<dataset>_<split>/<COCO annotation id>").

Items of the RefCOCO train split (lists with a refcoco_ref_id column) use the original COCO train2014 images
(images.cocodataset.org/train2014/) from a local directory, and the box of the list:

  python -m spred.items --list LIST.csv --coco-dir DIR --out work/items/training_extension.jsonl

with seed_key "refcoco_train/<RefCOCO ref id>".
"""
import argparse
import csv
import io
import json
import os
import re

from spred.geometry import xywh_to_xyxy

SEED_PREFIX = {("RefCOCO", "val"): "refcoco_val", ("RefCOCO", "testB"): "refcoco_testB", ("RefCOCOg (UMD)", "val"): "refcocog_val",
               ("RefCOCOg (UMD)", "test"): "refcocog_test", ("RefCOCO+", "testA"): "refcocoplus_testA", ("RefCOCO+", "val"): "refcocoplus_val"}


def parquet_files(repo, revision, split):
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo, repo_type="dataset", revision=revision)
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.join(root, f) for f in sorted(files)
                if f.endswith(".parquet") and re.match(rf"{re.escape(split)}-\d+-of-\d+\.parquet$", f)]
    return out


def coco_image_id(file_name):
    m = re.search(r"COCO_\w+?_0*(\d+)", file_name or "")
    return int(m.group(1)) if m else None


def from_coco(items, coco_dir, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    missing = 0
    with open(out, "w") as f:
        for it in items:
            path = os.path.join(coco_dir, f"COCO_train2014_{int(it['coco_image_id']):012d}.jpg")
            missing += not os.path.isfile(path)
            f.write(json.dumps({"id": it["item_id"], "image_id": int(it["coco_image_id"]), "img": path, "ref": it["expression"],
                                "box": [float(it[k]) for k in ("box_x1", "box_y1", "box_x2", "box_y2")],
                                "W": int(it["image_width"]), "H": int(it["image_height"]),
                                "seed_key": f"refcoco_train/{it['refcoco_ref_id']}"}, ensure_ascii=False) + "\n")
    print(f"[items] {len(items)} items -> {out}" + (f" ({missing} images not in {coco_dir})" if missing else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True); ap.add_argument("--dataset"); ap.add_argument("--revision")
    ap.add_argument("--split"); ap.add_argument("--image-dir"); ap.add_argument("--coco-dir"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.list, newline="", encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    if a.coco_dir:
        return from_coco(items, a.coco_dir, a.out)
    if not (a.dataset and a.revision and a.split and a.image_dir):
        ap.error("--dataset, --revision, --split and --image-dir are required unless --coco-dir is given")
    import pyarrow.parquet as pq
    from PIL import Image
    want = {(int(r["coco_image_id"]), r["expression"]) for r in items}
    rows = {}
    for p in parquet_files(a.dataset, a.revision, a.split):
        for r in pq.read_table(p).to_pylist():
            ref = r.get("answer")
            ref = ref[0] if isinstance(ref, list) and ref else ref
            key = (coco_image_id(r.get("file_name")), str(ref).strip() if ref else None)
            if key in want and key not in rows:
                rows[key] = r
    os.makedirs(a.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    missing = 0
    with open(a.out, "w") as f:
        for it in items:
            r = rows.get((int(it["coco_image_id"]), it["expression"]))
            if r is None:
                missing += 1
                continue
            img = r["image"]
            im = Image.open(io.BytesIO(img.get("bytes") if isinstance(img, dict) else img)).convert("RGB")
            path = os.path.join(a.image_dir, f"{it['coco_image_id']}.jpg")
            if not os.path.exists(path):
                im.save(path, quality=95)
            W, H = im.size
            prefix = SEED_PREFIX[(it["dataset"], it["split"])]
            f.write(json.dumps({"id": it["item_id"], "image_id": int(it["coco_image_id"]), "img": path, "ref": it["expression"],
                                "box": xywh_to_xyxy([float(v) for v in r["bbox"]]), "W": W, "H": H,
                                "seed_key": f"{prefix}/{it['coco_annotation_id']}"}, ensure_ascii=False) + "\n")
    print(f"[items] {len(items) - missing} of {len(items)} items -> {a.out}" + (f" ({missing} not found in the dataset)" if missing else ""))


if __name__ == "__main__":
    main()
