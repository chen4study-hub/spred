"""Item lists and item files for the two other image domains: the Objects365 part of Ref-L4 and VRSBench.

Both lists are drawn from the public annotation files with a fixed seed; whole images go to one side, the test items
are drawn first and the unlabelled training items (1,000) after them.
  objects365  Ref-L4 tables ref-l4-val.parquet and ref-l4-test.parquet (huggingface.co/datasets/JierunChen/Ref-L4);
              images with an Objects365 image id, original expressions only (no rewrites), boxes of at least 1% of the
              image, one expression per annotated box; images from images.tar.gz of the same dataset
  vrsbench    VRSBench_EVAL_referring.json (huggingface.co/datasets/xiang709/VRSBench); one expression per image, the
              box given in percent of the image is converted to pixels with the image size; images from Images_val.zip

  python -m spred.other_domains objects365 --tables DIR --image-dir DIR --lists-dir work/lists --items-dir work/items
  python -m spred.other_domains vrsbench --ann VRSBench_EVAL_referring.json --image-dir DIR --lists-dir work/lists --items-dir work/items

Writes <lists-dir>/<domain>_{test,training}.csv (columns as data/lists/objects365_test.csv; the test list equals the
one in data/lists/) and <items-dir>/<domain>_{test,training}.jsonl for spred.run_model (seed_key "<prefix>/<source id>").
For VRSBench the image directory must already hold the images (their size gives the pixel boxes).
"""
import argparse
import ast
import csv
import json
import os
import random
import re

GT = re.compile(r"\{<(\d+)><(\d+)><(\d+)><(\d+)>\}")
DOMAINS = {"objects365": ("objects365", "o365"), "vrsbench": ("vrsbench", "vrs")}


def select_objects365(rows, n_train=1000, n_test=1000, seed=0, min_area_frac=0.01):
    """rows of the Ref-L4 tables -> (test, training) lists of dicts with src, image_id, file_name, ref, box, W, H"""
    rng = random.Random(seed)
    keep = {}
    for r in sorted(rows, key=lambda r: str(r["id"])):
        if not str(r["image_id"]).startswith("o365") or str(r["is_rewrite"]) == "True":
            continue
        x, y, w, h = [float(v) for v in (ast.literal_eval(r["bbox"]) if isinstance(r["bbox"], str) else r["bbox"])]
        W, H = int(float(r["width"])), int(float(r["height"]))
        if w * h < min_area_frac * W * H:
            continue
        keep.setdefault(str(r["bbox_id"]), []).append(dict(src=str(r["id"]), image_id=str(r["image_id"]), file_name=r["file_name"],
                                                         ref=r["caption"], box=[x, y, x + w, y + h], W=W, H=H))
    items = [rng.choice(v) for _, v in sorted(keep.items())]
    by_img = {}
    for it in items:
        by_img.setdefault(it["image_id"], []).append(it)
    imgs = sorted(by_img)
    rng.shuffle(imgs)
    train, test = [], []
    for im in imgs:
        if len(test) < n_test:
            test += by_img[im]
        elif len(train) < n_train:
            train += by_img[im]
    return test[:n_test], train[:n_train]


def select_vrsbench(rows, n_train=1000, n_test=1000, seed=0):
    """rows of VRSBench_EVAL_referring.json -> (test, training) lists of dicts with src, image_id, file_name, ref, pct"""
    rng = random.Random(seed)
    by_img = {}
    for r in sorted(rows, key=lambda r: int(r["question_id"])):
        m = GT.fullmatch(r["ground_truth"].strip())
        if not m:
            continue
        p = [int(v) for v in m.groups()]
        if p[2] <= p[0] or p[3] <= p[1]:
            continue
        by_img.setdefault(r["image_id"], []).append(dict(src=str(r["question_id"]), image_id=os.path.splitext(r["image_id"])[0],
                                                         file_name=r["image_id"], ref=r["question"], pct=p))
    imgs = sorted(by_img)
    rng.shuffle(imgs)
    picked = [rng.choice(by_img[im]) for im in imgs]
    return picked[:n_test], picked[n_test:n_test + n_train]


def with_pixel_boxes(items, image_dir):
    from PIL import Image
    for it in items:
        with Image.open(os.path.join(image_dir, it["file_name"])) as im:
            W, H = im.size
        x1, y1, x2, y2 = it["pct"]
        it.update(W=W, H=H, box=[x1 * W / 100.0, y1 * H / 100.0, x2 * W / 100.0, y2 * H / 100.0])
    return items


def write(domain, test, train, image_dir, lists_dir, items_dir):
    name, prefix = DOMAINS[domain]
    os.makedirs(lists_dir, exist_ok=True)
    os.makedirs(items_dir, exist_ok=True)
    for part, lst, idfmt in (("test", test, f"{name}-{{:04d}}"), ("training", train, f"{name}-train-{{:04d}}")):
        seed_prefix = f"{prefix}_{'test' if part == 'test' else 'train'}"
        with open(os.path.join(lists_dir, f"{name}_{part}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "source_id", "image_id", "box_x1", "box_y1", "box_x2", "box_y2", "image_width", "image_height"])
            for k, it in enumerate(lst):
                w.writerow([idfmt.format(k + 1), it["src"], it["image_id"]] + [f"{v:.2f}" for v in it["box"]] + [it["W"], it["H"]])
        with open(os.path.join(items_dir, f"{name}_{part}.jsonl"), "w") as f:
            for k, it in enumerate(lst):
                f.write(json.dumps({"id": idfmt.format(k + 1), "image_id": it["image_id"], "img": os.path.join(image_dir, it["file_name"]),
                                    "ref": it["ref"], "box": it["box"], "W": it["W"], "H": it["H"],
                                    "seed_key": f"{seed_prefix}/{it['src']}"}, ensure_ascii=False) + "\n")
        miss = sum(not os.path.isfile(os.path.join(image_dir, it["file_name"])) for it in lst)
        print(f"[{domain}] {part}: {len(lst)} items on {len({it['image_id'] for it in lst})} images" + (f"; {miss} images not in {image_dir}" if miss else ""))
    assert not ({it["image_id"] for it in test} & {it["image_id"] for it in train}), "test and training lists share images"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="domain", required=True)
    p = sp.add_parser("objects365")
    p.add_argument("--tables", required=True)
    p = sp.add_parser("vrsbench")
    p.add_argument("--ann", required=True)
    for p in sp.choices.values():
        p.add_argument("--image-dir", required=True); p.add_argument("--lists-dir", required=True); p.add_argument("--items-dir", required=True)
    a = ap.parse_args()
    if a.domain == "objects365":
        import pyarrow.parquet as pq
        rows = []
        for f in ("ref-l4-val.parquet", "ref-l4-test.parquet"):
            rows += pq.read_table(os.path.join(a.tables, f)).to_pylist()
        test, train = select_objects365(rows)
    else:
        with open(a.ann) as f:
            test, train = select_vrsbench(json.load(f))
        test, train = with_pixel_boxes(test, a.image_dir), with_pixel_boxes(train, a.image_dir)
    write(a.domain, test, train, a.image_dir, a.lists_dir, a.items_dir)


if __name__ == "__main__":
    main()
