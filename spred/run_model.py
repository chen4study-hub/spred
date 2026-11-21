"""Model runs (GPU): written boxes, a shifted view, sampled answers and hidden-state features.

  generate  greedy answer with the official grounding prompt; with --shifted-view also the answer for the image
            padded on the left (two merged-token widths for the Qwen models, 64 pixels for the others), shifted back
            into the frame of the unchanged image (the shift teacher and the content-shift check)
  sample    one sampled answer per item (temperature 0.7, top-p 0.9, seed from the item's sampling key; --seed-offset
            draws another sample), the sampling teacher
  features  one teacher-forced forward pass over prompt + greedy answer: hidden state of the last prompt token and
            mean hidden state of the coordinate tokens at every fourth layer (and the last), mean log-probability of
            the coordinate tokens; for the Qwen models also the attention of the coordinate tokens to the image tokens
            inside the written box and to all image tokens, per layer and head (eager attention)

  python -m spred.run_model generate --model Qwen/Qwen2.5-VL-3B-Instruct --items work/items/training.jsonl \
      --out work/answers/training_qwen25vl3b.jsonl --shifted-view
  python -m spred.run_model sample   --model Qwen/Qwen2.5-VL-3B-Instruct --items work/items/training.jsonl \
      --out work/answers/sampled_training_qwen25vl3b.jsonl
  python -m spred.run_model features --model Qwen/Qwen2.5-VL-3B-Instruct --items work/items/training.jsonl \
      --answers work/answers/training_qwen25vl3b.jsonl --out work/features/training_qwen25vl3b.pt

Every command resumes an interrupted output file (features excepted) and takes --limit N for the first N items.
Answer files hold one JSON line per item: id, image_id, gold and the written box `pred` (pixels of the image shown),
its IoU, the generated token ids, and `pred_shifted_back` with --shifted-view.
"""
import argparse
import json
import os
import time

from spred import geometry as G
from spred import models


def resume(path):
    return {r["id"] for r in G.load_jsonl(path)} if os.path.exists(path) else set()


def cmd_generate(a):
    import torch
    run = models.load(a.model)
    items = G.load_jsonl(a.items)[: a.limit or None]
    done = resume(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0, n = time.time(), 0
    with open(a.out, "a") as f, torch.inference_mode():
        for it in items:
            if it["id"] in done:
                continue
            img, gold = run.image(it)
            gen, text = run.answer(run.inputs(img, it["ref"]), run.max_new)
            pred = run.box(text, img)
            rec = {"id": it["id"], "image_id": it["image_id"], "gold": gold, "W": img.size[0], "H": img.size[1],
                   "text": text[:300], "gen_ids": gen, "pred": pred, "iou": G.iou(pred, gold) if pred else None}
            if a.shifted_view:
                pimg = models.pad_left(img, run.shift_pad)
                _, ptext = run.answer(run.inputs(pimg, it["ref"]), run.max_new)
                pp = run.box(ptext, pimg)
                rec["pred_shifted_back"] = G.shift_box(pp, dx=-run.shift_pad) if pp else None
            f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 100 == 0:
                f.flush()
                print(f"[generate] {n} {time.time() - t0:.0f}s ({(time.time() - t0) / n:.2f} s/item)", flush=True)
    print(f"[generate] {n} items in {time.time() - t0:.0f}s -> {a.out}")


def cmd_sample(a):
    import torch
    run = models.load(a.model)
    items = G.load_jsonl(a.items)[: a.limit or None]
    done = resume(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0, n = time.time(), 0
    with open(a.out, "a") as f, torch.inference_mode():
        for it in items:
            if it["id"] in done:
                continue
            img, _ = run.image(it)
            _, text = run.answer(run.inputs(img, it["ref"]), run.max_new, seed=G.item_seed(it["seed_key"], a.seed_offset))
            f.write(json.dumps({"id": it["id"], "box": run.box(text, img), "text": text[:300]}) + "\n")
            n += 1
            if n % 100 == 0:
                f.flush()
                print(f"[sample] {n} {time.time() - t0:.0f}s ({(time.time() - t0) / n:.2f} s/item)", flush=True)
    print(f"[sample] {n} items in {time.time() - t0:.0f}s -> {a.out}")


def cmd_features(a):
    import torch
    run = models.load(a.model, attn="eager")
    layers = sorted(set(range(0, run.n_layers + 1, a.layer_step)) | {run.n_layers})
    items = {it["id"]: it for it in G.load_jsonl(a.items)}
    answers = [g for g in G.load_jsonl(a.answers) if g["pred"] is not None][: a.limit or None]
    H_last, H_coord, logp, att_in, att_img, ids = [], [], [], [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for k, g in enumerate(answers):
            it = items[g["id"]]
            img, _ = run.image(it)
            inp = run.inputs(img, it["ref"])
            P = inp["input_ids"].shape[1]
            full = torch.cat([inp["input_ids"], torch.tensor([g["gen_ids"]], device="cuda")], 1)
            ci = G.coord_token_idx([run.proc.tokenizer.decode([t]) for t in g["gen_ids"]], after=run.coord_after)
            if not ci:
                continue
            kw = dict(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=inp["pixel_values"], output_hidden_states=True)
            if run.attention:
                pos, _ = run.model.model.get_rope_index(full, inp["image_grid_thw"], None, attention_mask=torch.ones_like(full))
                s, e = G.image_spans(full[0].tolist(), run.image_token_id)[0]
                rows = (pos[1, 0, s:e] - pos[1, 0, s:e].min()).tolist()
                cols = (pos[2, 0, s:e] - pos[2, 0, s:e].min()).tolist()
                mask = torch.tensor(G.cells_in_box(rows, cols, run.factor, g["pred"]), device="cuda", dtype=torch.float32)
                run.model.model.rope_deltas = None
                kw.update(image_grid_thw=inp["image_grid_thw"], output_attentions=True)
            out = run.model(**kw)
            cpos = [P + i for i in ci]
            H_last.append(torch.stack([out.hidden_states[l][0, P - 1] for l in layers]).half().cpu())
            H_coord.append(torch.stack([out.hidden_states[l][0, cpos].float().mean(0) for l in layers]).half().cpu())
            lp = torch.log_softmax(out.logits[0, [c - 1 for c in cpos]].float(), -1)
            logp.append(lp.gather(1, full[0, cpos].unsqueeze(1)).mean().item())
            if run.attention:
                ain, aimg = [], []
                for A in out.attentions:                  # per layer: (1, heads, L, L)
                    r = A[0, :, cpos, s:e].float()       # (heads, coordinate tokens, image tokens)
                    ain.append((r * mask).sum(-1).mean(-1))
                    aimg.append(r.sum(-1).mean(-1))
                att_in.append(torch.stack(ain).cpu())
                att_img.append(torch.stack(aimg).cpu())
            ids.append(g["id"])
            del out
            if (k + 1) % 100 == 0:
                print(f"[features] {k + 1}/{len(answers)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    z = torch.zeros(len(ids), 1)
    torch.save({"ids": ids, "layers": layers, "h_last": torch.stack(H_last), "h_coord": torch.stack(H_coord), "logp": torch.tensor(logp),
                "att_in": torch.stack(att_in) if att_in else z, "att_img": torch.stack(att_img) if att_img else z.clone()}, a.out)
    print(f"[features] {len(ids)} of {len(answers)} items in {time.time() - t0:.0f}s -> {a.out}; layers {layers}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "sample", "features"):
        p = sp.add_parser(name)
        p.add_argument("--model", required=True); p.add_argument("--items", required=True); p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int, default=0)
        if name == "generate":
            p.add_argument("--shifted-view", action="store_true")
        if name == "sample":
            p.add_argument("--seed-offset", type=int, default=0)
        if name == "features":
            p.add_argument("--answers", required=True); p.add_argument("--layer-step", type=int, default=4)
    a = ap.parse_args()
    {"generate": cmd_generate, "sample": cmd_sample, "features": cmd_features}[a.cmd](a)


if __name__ == "__main__":
    main()
