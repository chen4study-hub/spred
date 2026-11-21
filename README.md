# SPRED: self-prediction residuals for wrong-object boxes

Code and data for *Self-prediction residuals improve single-pass detection of wrong-object boxes in grounded
vision-language models*. A grounded VLM writes a box for a referring expression. A ridge read-out, fitted on the
model's own unlabelled answers, reads a box from the hidden state of the last prompt token; the self-prediction
residual is 1 − IoU between the two boxes. SPRED adds it to a label-free two-teacher probe:
z(probe) + 0.5 z(residual). Models: Qwen2.5-VL-3B / 7B, Qwen3-VL-4B / 8B, InternVL2.5-8B-MPO, LLaVA-1.5-7B.

## Installation

Reproducing the results needs Python 3 with numpy only. Running the models (Python 3.12, one 32 GB GPU):

```
pip install -r requirements.txt     # torch 2.5.1 (CUDA 12.4), transformers 4.57.6
```

## Data

`data/` holds the item lists, per-item answers and scores, and the scripts that recompute the reported numbers: the
item lists, written boxes and per-item scores of every item set, standardization constants and refit gains (`data/scores/`, `data/answers/`, `data/lists/`, `data/refits/`),
further per-item data behind the main-text tables and figures (`data/results/`: boxes in relative coordinates with
the box of the shifted view, development sets of Qwen2.5-VL-3B, BICR scores on the replication and held-out lists
with its training constants, the read-out error sigma per model and fit, COCO instances of the images), the
standardization constants of the fits on the two other image domains, the further RefCOCO training items of the
matched-size fits (`data/lists/training_extension.csv`), and the evaluation scripts (`data/code/`). Every file and
column is described in `data/README.md`.

No images, hidden states or model weights are included. Images: MS-COCO train2014 through the Hugging Face copies of
RefCOCO (`lmms-lab-encoder/RefCOCO`, revision `a5dff0b31947`), RefCOCOg (`lmms-lab-encoder/RefCOCOg`,
`93af55545967`) and RefCOCO+ (`lmms-lab/RefCOCOplus`, `a283082593c4`); Objects365 images from Ref-L4
(`JierunChen/Ref-L4`, `images.tar.gz`); remote-sensing images from VRSBench (`xiang709/VRSBench`, `Images_val.zip`).
Models: `Qwen/Qwen2.5-VL-3B-Instruct`,
`Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`,
`OpenGVLab/InternVL2_5-8B-MPO-hf`, `llava-hf/llava-1.5-7b-hf` (revisions in Section S1 of the supplementary material).

## Reproducing the results

The commands below recompute the numbers of the main-text tables and figures. Every number they check is listed with
its place in the paper in `data/code/expected.csv`, `data/results/expected_main.csv` and `data/code/expected_matched.csv`.
Run them from `data/`:

```
cd data
python code/evaluate.py check          # 582 numbers
python code/reproduce_main.py check    # 444 numbers
python code/matched_sizes.py check     # 38 numbers
```

Each prints every number with its place in the paper, the recomputed value and the reported value, and exits with
an error if any differs after rounding. Resampling is fixed (seed 0), so the intervals are reproduced exactly.

| Paper | Command (in `data/`) |
|---|---|
| Table 2 (lists) | `python code/reproduce_main.py lists` |
| Table 3; Figure 2, data (gain from adding the residual) | `python code/evaluate.py gain --set replication --errors wrong_object` (and `--set heldout`) |
| Table 4 (SPRED against prior scores) | `python code/evaluate.py table4 --set replication`, `counts`, `review --share 0.2`; BICR and per-model review rows: `python code/reproduce_main.py table4` |
| Table 5 (ablation) | `python code/reproduce_main.py ablation` |
| Section 3.6 (read-out error; written boxes narrower than twice it) | `python code/reproduce_main.py narrow` |
| Section 5.2 (training sizes up to 10,000 items) | `python code/matched_sizes.py table` |
| Figure 3, data (second answer and read-out) | `python code/reproduce_main.py second-answer` |
| Figure 4, data (read-out displacement) | `python code/reproduce_main.py pull` |
| Figure 5 (examples) | the examples are items `new-testB-0088`, `new-refcocog-0348`, `new-testB-0550`, `new-testB-0310`, `new-testB-0318` and `new-testB-0119` of `scores/new-items_Qwen2.5-VL-3B.csv` (COCO images 439906, 466519, 471315, 39395, 397423, 192476) |
| Abstract, highlights | `python code/evaluate.py counts`, `gain --errors all`, `review --set replication --share 0.2` |
| Section 5.2 (models on which the probe alone and SPRED come within 0.03 of the supervised probe) | `python code/reproduce_main.py margin` |
| Section 3.3 (accuracy of the read-out box and of the written box) | `python code/reproduce_main.py readout-accuracy` |
| Introduction, Section 6.3 (wrong-object errors on another annotated object) | `python code/reproduce_main.py instances` |
| Section 3.6, Section 7 (Objects365 and VRSBench; gain by target width) | `python code/reproduce_main.py domains` |
| Introduction, Section 7 (which of two models erred, replication list) | `python code/evaluate.py pairs --set replication` |
| Section 5.1, Table S6.8 (wrong-object errors against the other errors, among wrong answers) | `python code/evaluate.py types --set replication` |

`data/README.md` lists the files and columns behind each number.

## Running the pipeline

The per-item scores are produced in three steps (`work/` is a scratch directory; GPU steps need the model weights).
`--model` of the GPU steps (`spred.run_model`, `spred.run_points`, `spred.bicr blank-states`) takes the Hugging Face
id, e.g. `Qwen/Qwen2.5-VL-3B-Instruct`; `--model-name` and `--models` of the CPU steps take the short name used in
`data/`, e.g. `Qwen2.5-VL-3B`, which also names their output files. Below, M stands for the form the option takes.

```
# 1. item files with local image copies, one per list (here the training items; the test lists likewise)
python -m spred.items --list data/lists/training.csv --dataset lmms-lab-encoder/RefCOCO --revision a5dff0b31947 \
    --split val --image-dir work/images --out work/items/training.jsonl

#    the further RefCOCO train items of the matched-size fits use the original COCO train2014 images:
python -m spred.items --list data/lists/training_extension.csv --coco-dir DIR --out work/items/training_extension.jsonl

# 2. model runs (GPU), for each model M and list L; sampled answers on the training items only
python -m spred.run_model generate --model M --items work/items/L.jsonl --out work/answers/L_M.jsonl --shifted-view
python -m spred.run_model features --model M --items work/items/L.jsonl --answers work/answers/L_M.jsonl --out work/features/L_M.pt
python -m spred.run_model sample   --model M --items work/items/training.jsonl --out work/answers/sampled_training_M.jsonl
python -m spred.run_model sample   --model M --items work/items/training.jsonl --out work/answers/sampled2_training_M.jsonl --seed-offset 1

# 3. fit on the training items and score the test lists (CPU, OMP_NUM_THREADS=4)
python -m spred.fit_scores --model-name M --train work/answers/training_M.jsonl work/features/training_M.pt \
    --train-samples work/answers/sampled_training_M.jsonl \
    --test replication work/answers/replication_M.jsonl work/features/replication_M.pt --out-dir work/scores/fit0
```

Step 3 writes the per-item scores, the standardization constants and the selected configurations of the reported
fit, including the read-out error sigma; `--refit K` (K = 1 … 10, with `--train-samples-2` for the two Qwen2.5-VL
models) gives the refits on resampled training items behind the combined intervals.

Training sizes up to 10,000 items (Section 5.2, Tables S4.3 to S4.5): training size n uses the first n rows of
`data/lists/training.csv` followed by `data/lists/training_extension.csv`, in this order (for Qwen2.5-VL-7B without
the 3 items of the extension that have no parsed answer). Run step 2 (`generate --shifted-view`, `sample`,
`features`) on the extension items as on the training items, join the answer files, the sampled-answer files and the
feature files of the two parts in this order (the tensors of a feature file stacked along the first axis, `ids`
concatenated), keep the first n rows and pass them as `--train` and `--train-samples` of `spred.fit_scores`; the
supervised probes are then searched at that size. `spred.bicr blank-states` and `spred.bicr fit` take the joined
files in the same way.

The perceptron probe is trained in single precision and depends slightly on the number of threads. The review with
unequal costs (the cost ratio of a wrong-object error at which SPRED catches more than the probe) uses the stored fit
and the ten refits:

```
python -m spred.review_cost --set replication --list data/lists/replication.csv --models M1 M2 ... \
    --fits work/scores/fit0 work/scores/refit1 ... work/scores/refit10
```

**Point answers** (Qwen models). The model is asked for one point instead of a box; the training items are the first
1,000 items of `data/lists/training.csv`:

```
python -m spred.run_points generate --model M --prompt plain --items work/items/training.jsonl --limit 1000 \
    --shifted-view --out work/points/training_M.jsonl        # --prompt keyed for Qwen2.5-VL-7B and Qwen3-VL-4B
python -m spred.run_points sample   --model M --prompt plain --items work/items/training.jsonl --limit 1000 --out work/points/sampled_training_M.jsonl
python -m spred.run_points features --model M --prompt plain --items work/items/training.jsonl --answers work/points/training_M.jsonl \
    --out work/points/training_M.pt                          # generate and features likewise for the test lists
python -m spred.fit_points --model-name M --train work/points/training_M.jsonl work/points/training_M.pt \
    --train-samples work/points/sampled_training_M.jsonl \
    --test replication work/points/replication_M.jsonl work/points/replication_M.pt --out-dir work/points/fit0
```

**Other image domains.** The Objects365 and VRSBench lists (test items as in `data/lists/`, and 1,000 unlabelled
training items each) are drawn from the public annotation files; steps 2 and 3 then run as above, with the training
items of the domain or with the RefCOCO training items:

```
python -m spred.other_domains objects365 --tables DIR --image-dir DIR --lists-dir work/lists --items-dir work/items
python -m spred.other_domains vrsbench --ann VRSBench_EVAL_referring.json --image-dir DIR --lists-dir work/lists --items-dir work/items
python -m spred.target_size --fit-dir work/scores/objects365_in-domain --set objects365 --list work/lists/objects365_test.csv \
    --models Qwen2.5-VL-3B Qwen2.5-VL-7B Qwen3-VL-4B         # targets narrower than 2 sigma and the gain by target size
```

**BICR baseline** (our reimplementation): hidden states of the training items with the image replaced by a black
image, then the probe with its blind-image ranking term, configuration chosen by out-of-fold AUROC:

```
python -m spred.bicr blank-states --model M --items work/items/training.jsonl --answers work/answers/training_M.jsonl \
    --out work/features/blank_training_M.pt
python -m spred.bicr fit --train work/answers/training_M.jsonl work/features/training_M.pt \
    --blank work/features/blank_training_M.pt --test replication work/features/replication_M.pt --out-dir work/scores/bicr
```

## Analyses provided as per-item scores

Some analyses of the supplementary material use additional generation passes or an external detector. For these,
`data/` holds the per-item scores, and the reported numbers are computed from them; the code that produces the
scores is not part of `spred/`:

- the disagreement of the box over seven perturbed views (`seven_view_disagreement` in `data/scores/new-items_*.csv`);
- the open-vocabulary detector used as a second localizer (`external_top_box_disagreement`, `external_support`);
- the fits on 5,000 and 10,000 training items (`data/scores/matched-sizes/`; the further training items are listed
  in `data/lists/training_extension.csv`).

The columns are described in `data/README.md`.

A few numbers of the paper need material that is neither in `data/` nor produced by `spred/`: the share of errors
whose box all seven perturbed views keep, and the AUROC of SPRED on these items (Introduction, Section 7, Table S7.8;
`data/` holds the mean disagreement over the views, not the box of each view); the share of wrong-object errors on
annotations that Ref-L4 found faulty (Section 6.1; it uses the annotation checks of Ref-L4); the run time of SPRED
relative to generation alone (Section 3.6; measured on the GPU); and the fits on subsets of 250 labelled training
items and with one error type left out of the training items (Section 5.3, Tables S7.1 and S7.6; the subsets are
not listed).

## Structure

```
spred/   items.py, other_domains.py        item files and item lists
         run_model.py, run_points.py       model runs: boxes, points (GPU)
         models.py, geometry.py            Qwen / InternVL / LLaVA loading, prompts, answer parsing, coordinates
         fit_scores.py, fit_points.py      read-out, probes and reference scores (CPU)
         probes.py, bicr.py, target_size.py, review_cost.py
data/    per-item data, data/results/, evaluation scripts in data/code/
```

## License

The code is released under the Apache License 2.0 (`LICENSE`); the data in `data/` under CC BY 4.0
(`data/LICENSE`). The item lists refer to RefCOCO, RefCOCO+, RefCOCOg, MS-COCO, Ref-L4 (Objects365 images) and
VRSBench, which remain under their own licenses; no images are included. `data/results/coco_instances.json` is
derived from the MS-COCO 2014 annotations (CC BY 4.0).
