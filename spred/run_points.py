"""Point answers (GPU, Qwen models): the model is asked for one point on the referred object instead of a box.

  generate  greedy point; with --shifted-view also the point for the image padded on the left by two merged-token
            widths, shifted back into the frame of the unchanged image
  sample    one sampled point per item (temperature 0.7, top-p 0.9, seed from the item's sampling key)
  features  one teacher-forced forward pass over prompt + greedy answer: hidden state of the last prompt token and mean
            hidden state of the tokens that carry the digits of the two coordinates, at every fourth layer (and the
            last); mean log-probability of these tokens. Answers whose two numbers cannot be separated are skipped.

Prompts (--prompt): 'plain' was used for Qwen2.5-VL-3B, 'keyed' for Qwen2.5-VL-7B and Qwen3-VL-4B. The training items
of the point answers are the first 1,000 items of data/lists/training.csv (--limit 1000), with the shifted view and
one sampled answer; the test lists are answered without them.

  python -m spred.run_points generate --model Qwen/Qwen2.5-VL-3B-Instruct --prompt plain --items work/items/training.jsonl \
      --limit 1000 --shifted-view --out work/points/training_qwen25vl3b.jsonl

Answer files hold one JSON line per item: id, image_id, gold (annotated box in pixels of the image shown), the point
`pt`, `hit` (point inside the annotated box), the generated token ids, and `pt_shifted_back` with --shifted-view.
"""
import argparse
import json
import os
import time

from spred import geometry as G
from spred import models


def load(a, attn="sdpa"):
    if models.family_of(a.model) != "qwen":
        raise SystemExit("point answers are run on the Qwen models")
    run = models.QwenRunner(a.model, attn)
    run.template = models.POINT_PROMPTS[a.prompt]
    return run


def resume(path):
    return {r["id"] for r in G.load_jsonl(path)} if os.path.exists(path) else set()


def answer(run, img, ref, max_new, seed=None):
    gen, text = run.answer(run.inputs(img, ref), max_new, seed=seed)
    return run.point(text, img), text, gen


def cmd_generate(a):
    import torch
    run = load(a)
    items = G.load_jsonl(a.items)[: a.limit or None]
    done = resume(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0, n = time.time(), 0
    with open(a.out, "a") as f, torch.inference_mode():
        for it in items:
            if it["id"] in done:
                continue
            img, gold = run.image(it)
            pt, text, gen = answer(run, img, it["ref"], a.max_new)
            rec = {"id": it["id"], "image_id": it["image_id"], "gold": gold, "W": img.size[0], "H": img.size[1],
                   "text": text[:300], "gen_ids": gen, "pt": pt, "hit": bool(G.inside(pt, gold)) if pt else None}
            if a.shifted_view:
                pimg = models.pad_left(img, run.shift_pad)
                pp, _, _ = answer(run, pimg, it["ref"], a.max_new)
                rec["pt_shifted_back"] = [pp[0] - run.shift_pad, pp[1]] if pp else None
            f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 100 == 0:
                f.flush()
                print(f"[generate] {n} {time.time() - t0:.0f}s ({(time.time() - t0) / n:.2f} s/item)", flush=True)
    print(f"[generate] {n} items in {time.time() - t0:.0f}s -> {a.out}")


def cmd_sample(a):
    import torch
    run = load(a)
    items = G.load_jsonl(a.items)[: a.limit or None]
    done = resume(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0, n = time.time(), 0
    with open(a.out, "a") as f, torch.inference_mode():
        for it in items:
            if it["id"] in done:
                continue
            img, _ = run.image(it)
            pt, text, _ = answer(run, img, it["ref"], a.max_new, seed=G.item_seed(it["seed_key"]))
            f.write(json.dumps({"id": it["id"], "pt": pt, "text": text[:300]}) + "\n")
            n += 1
            if n % 100 == 0:
                f.flush()
                print(f"[sample] {n} {time.time() - t0:.0f}s ({(time.time() - t0) / n:.2f} s/item)", flush=True)
    print(f"[sample] {n} items in {time.time() - t0:.0f}s -> {a.out}")


def cmd_features(a):
    import torch
    run = load(a)
    layers = sorted(set(range(0, run.n_layers + 1, a.layer_step)) | {run.n_layers})
    items = {it["id"]: it for it in G.load_jsonl(a.items)}
    answers = [g for g in G.load_jsonl(a.answers) if g["pt"] is not None][: a.limit or None]
    HL, HC, logp, ids, skipped = [], [], [], [], 0
    t0 = time.time()
    with torch.inference_mode():
        for k, g in enumerate(answers):
            groups = G.point_number_tokens([run.proc.tokenizer.decode([t]) for t in g["gen_ids"]])
            if groups is None:
                skipped += 1
                continue
            it = items[g["id"]]
            img, _ = run.image(it)
            inp = run.inputs(img, it["ref"])
            P = inp["input_ids"].shape[1]
            full = torch.cat([inp["input_ids"], torch.tensor([g["gen_ids"]], device=inp["input_ids"].device)], 1)
            run.model.model.rope_deltas = None
            out = run.model(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=inp["pixel_values"],
                            image_grid_thw=inp["image_grid_thw"], output_hidden_states=True)
            cpos = [P + i for grp in groups for i in grp]
            HL.append(torch.stack([out.hidden_states[l][0, P - 1] for l in layers]).half().cpu())
            HC.append(torch.stack([out.hidden_states[l][0, cpos].float().mean(0) for l in layers]).half().cpu())
            lp = torch.log_softmax(out.logits[0, [c - 1 for c in cpos]].float(), -1)
            logp.append(lp.gather(1, full[0, cpos].unsqueeze(1)).mean().item())
            ids.append(g["id"])
            if (k + 1) % 100 == 0:
                print(f"[features] {k + 1}/{len(answers)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"ids": ids, "layers": layers, "h_last": torch.stack(HL), "h_coord": torch.stack(HC), "logp": torch.tensor(logp)}, a.out)
    print(f"[features] {len(ids)} items ({skipped} without two separable numbers) in {time.time() - t0:.0f}s -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "sample", "features"):
        p = sp.add_parser(name)
        p.add_argument("--model", required=True); p.add_argument("--prompt", choices=sorted(models.POINT_PROMPTS), required=True)
        p.add_argument("--items", required=True); p.add_argument("--out", required=True); p.add_argument("--limit", type=int, default=0)
        if name == "features":
            p.add_argument("--answers", required=True); p.add_argument("--layer-step", type=int, default=4)
        else:
            p.add_argument("--max-new", type=int, default=48)
        if name == "generate":
            p.add_argument("--shifted-view", action="store_true")
    a = ap.parse_args()
    {"generate": cmd_generate, "sample": cmd_sample, "features": cmd_features}[a.cmd](a)


if __name__ == "__main__":
    main()
