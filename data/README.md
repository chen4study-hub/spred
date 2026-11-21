# Per-item data and evaluation code

Item lists, written boxes and per-item scores for "Self-prediction residuals improve single-pass detection of wrong-object boxes in grounded vision-language models". No images are included; the RefCOCO-family images are public MS-COCO images (train2014) identified by their COCO image id, and the images of the two other domains are identified by their ids in Ref-L4 and VRSBench.

The per-item scores and the evaluation code (`code/evaluate.py`, which recomputes the numbers listed in `code/expected.csv` from these scores, with `code/reproduce_main.py` and `code/matched_sizes.py` for further numbers) are provided in this directory. The code for running the models, extracting the features and fitting the read-out and the probes is in the public code repository https://github.com/chen4study-hub/spred (Apache-2.0).

## Files

| File | Rows | Content |
|---|---|---|
| `lists/training.csv` | 2000 | training items (the read-out and the label-free probes are fitted on them without labels; the supervised probes use their gold boxes) (1684 images) |
| `lists/new-items_refcoco-testB.csv` | 552 | new items, RefCOCO testB part (268 images) |
| `lists/new-items_refcocog-val.csv` | 600 | new items, RefCOCOg validation part (579 images) |
| `lists/refcocoplus_testA.csv` | 600 | RefCOCO+ testA items (267 images) |
| `lists/refcocoplus_val.csv` | 600 | RefCOCO+ validation items (445 images) |
| `lists/heldout.csv` | 1200 | held-out list (983 images) |
| `lists/replication.csv` | 1200 | replication list (901 images) |
| `lists/training_extension.csv` | 8000 | RefCOCO train items added to the training items for the training sizes up to 10,000 (Qwen2.5-VL-3B and Qwen2.5-VL-7B) (6501 images) |
| `lists/objects365_test.csv` | 1000 | Objects365 test items of Ref-L4 (793 images) |
| `lists/vrsbench_test.csv` | 1000 | VRSBench test items (remote-sensing images) (1000 images) |
| `lists/heldout_refcocog_test.jsonl` | 1200 | held-out list in JSON-lines form |
| `lists/replication_refcocog_test.jsonl` | 1200 | replication list in JSON-lines form |
| `answers/training_Qwen2.5-VL-3B.csv` | 2000 | written boxes of Qwen2.5-VL-3B on the training items (245 errors) |
| `answers/training_Qwen2.5-VL-7B.csv` | 2000 | written boxes of Qwen2.5-VL-7B on the training items (216 errors) |
| `answers/training_Qwen3-VL-4B.csv` | 2000 | written boxes of Qwen3-VL-4B on the training items (192 errors) |
| `answers/training_Qwen3-VL-8B.csv` | 1999 | written boxes of Qwen3-VL-8B on the training items (181 errors) |
| `answers/training_InternVL2.5-8B-MPO.csv` | 1997 | written boxes of InternVL2.5-8B-MPO on the training items (150 errors) |
| `answers/training_LLaVA-1.5-7B.csv` | 2000 | written boxes of LLaVA-1.5-7B on the training items (578 errors) |
| `scores/new-items_Qwen2.5-VL-3B.csv` | 1152 | Qwen2.5-VL-3B on the new items: written boxes, IoU and per-item scores (168 errors) |
| `scores/new-items_Qwen2.5-VL-7B.csv` | 1152 | Qwen2.5-VL-7B on the new items: written boxes, IoU and per-item scores (160 errors) |
| `scores/new-items_Qwen3-VL-4B.csv` | 1151 | Qwen3-VL-4B on the new items: written boxes, IoU and per-item scores (138 errors) |
| `scores/new-items_Qwen3-VL-8B.csv` | 1149 | Qwen3-VL-8B on the new items: written boxes, IoU and per-item scores (126 errors) |
| `scores/new-items_InternVL2.5-8B-MPO.csv` | 1152 | InternVL2.5-8B-MPO on the new items: written boxes, IoU and per-item scores (138 errors) |
| `scores/new-items_LLaVA-1.5-7B.csv` | 1152 | LLaVA-1.5-7B on the new items: written boxes, IoU and per-item scores (424 errors) |
| `scores/refcocoplus_Qwen2.5-VL-3B.csv` | 1200 | Qwen2.5-VL-3B on the RefCOCO+ items: written boxes, IoU and per-item scores (183 errors) |
| `scores/refcocoplus_Qwen2.5-VL-7B.csv` | 1200 | Qwen2.5-VL-7B on the RefCOCO+ items: written boxes, IoU and per-item scores (152 errors) |
| `scores/refcocoplus_InternVL2.5-8B-MPO.csv` | 1196 | InternVL2.5-8B-MPO on the RefCOCO+ items: written boxes, IoU and per-item scores (125 errors) |
| `scores/heldout_Qwen2.5-VL-3B.csv` | 1200 | Qwen2.5-VL-3B on the held-out list: written boxes, IoU and per-item scores (178 errors) |
| `scores/heldout_Qwen2.5-VL-7B.csv` | 1199 | Qwen2.5-VL-7B on the held-out list: written boxes, IoU and per-item scores (152 errors) |
| `scores/heldout_Qwen3-VL-4B.csv` | 1200 | Qwen3-VL-4B on the held-out list: written boxes, IoU and per-item scores (140 errors) |
| `scores/heldout_Qwen3-VL-8B.csv` | 1199 | Qwen3-VL-8B on the held-out list: written boxes, IoU and per-item scores (130 errors) |
| `scores/heldout_InternVL2.5-8B-MPO.csv` | 1199 | InternVL2.5-8B-MPO on the held-out list: written boxes, IoU and per-item scores (142 errors) |
| `scores/heldout_LLaVA-1.5-7B.csv` | 1200 | LLaVA-1.5-7B on the held-out list: written boxes, IoU and per-item scores (480 errors) |
| `scores/replication_Qwen2.5-VL-3B.csv` | 1200 | Qwen2.5-VL-3B on the replication list: written boxes, IoU and per-item scores (176 errors) |
| `scores/replication_Qwen2.5-VL-7B.csv` | 1199 | Qwen2.5-VL-7B on the replication list: written boxes, IoU and per-item scores (152 errors) |
| `scores/replication_Qwen3-VL-4B.csv` | 1198 | Qwen3-VL-4B on the replication list: written boxes, IoU and per-item scores (149 errors) |
| `scores/replication_Qwen3-VL-8B.csv` | 1197 | Qwen3-VL-8B on the replication list: written boxes, IoU and per-item scores (132 errors) |
| `scores/replication_InternVL2.5-8B-MPO.csv` | 1197 | InternVL2.5-8B-MPO on the replication list: written boxes, IoU and per-item scores (127 errors) |
| `scores/replication_LLaVA-1.5-7B.csv` | 1200 | LLaVA-1.5-7B on the replication list: written boxes, IoU and per-item scores (457 errors) |
| `scores/objects365_fitted-in-domain_Qwen2.5-VL-3B.csv` | 1000 | Qwen2.5-VL-3B on the Objects365 part of Ref-L4 test items, read-out and probes fitted on unlabelled training items of this domain: written boxes, IoU and per-item scores (211 errors) |
| `scores/objects365_fitted-in-domain_Qwen2.5-VL-7B.csv` | 996 | Qwen2.5-VL-7B on the Objects365 part of Ref-L4 test items, read-out and probes fitted on unlabelled training items of this domain: written boxes, IoU and per-item scores (193 errors) |
| `scores/objects365_fitted-in-domain_Qwen3-VL-4B.csv` | 999 | Qwen3-VL-4B on the Objects365 part of Ref-L4 test items, read-out and probes fitted on unlabelled training items of this domain: written boxes, IoU and per-item scores (154 errors) |
| `scores/objects365_fitted-on-RefCOCO_Qwen2.5-VL-3B.csv` | 1000 | Qwen2.5-VL-3B on the Objects365 part of Ref-L4 test items, read-out and probes fitted on RefCOCO and applied unchanged: written boxes, IoU and per-item scores (211 errors) |
| `scores/objects365_fitted-on-RefCOCO_Qwen2.5-VL-7B.csv` | 996 | Qwen2.5-VL-7B on the Objects365 part of Ref-L4 test items, read-out and probes fitted on RefCOCO and applied unchanged: written boxes, IoU and per-item scores (193 errors) |
| `scores/objects365_fitted-on-RefCOCO_Qwen3-VL-4B.csv` | 999 | Qwen3-VL-4B on the Objects365 part of Ref-L4 test items, read-out and probes fitted on RefCOCO and applied unchanged: written boxes, IoU and per-item scores (154 errors) |
| `scores/vrsbench_fitted-in-domain_Qwen2.5-VL-3B.csv` | 1000 | Qwen2.5-VL-3B on the VRSBench test items, read-out and probes fitted on unlabelled training items of this domain: written boxes, IoU and per-item scores (526 errors) |
| `scores/vrsbench_fitted-in-domain_Qwen3-VL-4B.csv` | 997 | Qwen3-VL-4B on the VRSBench test items, read-out and probes fitted on unlabelled training items of this domain: written boxes, IoU and per-item scores (385 errors) |
| `scores/vrsbench_fitted-on-RefCOCO_Qwen2.5-VL-3B.csv` | 1000 | Qwen2.5-VL-3B on the VRSBench test items, read-out and probes fitted on RefCOCO and applied unchanged: written boxes, IoU and per-item scores (526 errors) |
| `scores/vrsbench_fitted-on-RefCOCO_Qwen3-VL-4B.csv` | 997 | Qwen3-VL-4B on the VRSBench test items, read-out and probes fitted on RefCOCO and applied unchanged: written boxes, IoU and per-item scores (385 errors) |
| `standardization.csv` | 92 | per model and exported single-pass score: number of training rows, missing values among them, and the mean and (population) standard deviation of the non-missing training values from which the score is standardized; a sum such as the detector is z(score) + 0.5 z(residual) with these constants |
| `refits/plugin_gain_refits.csv` | 1212 | gain in AUROC when 0.5 z(residual) is added to each score, by item set, error type and model and as the mean over the models: stored fit and the ten refits on training items redrawn with replacement |
| `code/evaluate.py` | | evaluation script (Python 3 with numpy): AUROC, gains with combined intervals, errors caught under review, the read-out on wrong-object errors |
| `code/expected.csv` | 582 | the numbers of the paper that `evaluate.py check` recomputes: place in the paper, quantity, value, decimals |
| `refits/geometry_classifier_gain_refits.csv` | 21 | gain when the residual is added to the classifier on box geometry, among wrong answers, by item set and model and as the mean over the models: stored fit and ten refits of the two classifiers on redrawn training errors |
| `standardization_other_domains.csv` | 75 | per domain (Objects365, VRSBench), model and single-pass score of the fit on unlabelled training items of that domain: number of training rows, missing values among them, and the mean and (population) standard deviation of the non-missing training values; the scores and sums in `scores/<domain>_fitted-in-domain_<model>.csv` are standardized with these constants |
| `standardization_others_residual.csv` | 6 | per model: mean and (population) standard deviation, on the training items, of the mean standardized residual of the other five models; used to standardize that mean on the test items |
| `refits/others_residual_gain_refits.csv` | 60 | gain on wrong-object errors when the own residual or the mean residual of the other models is added to each score, mean over the models, replication and held-out list: stored fit and the ten refits on training items redrawn with replacement |
| `refits/pair_test_refits.csv` | 4 | share of discordant model pairs in which a score picks the model that erred (new items), for errors of any type and for wrong-object errors only, residual and detector: stored fit and the ten refits |
| `scores/points_replication_Qwen2.5-VL-3B.csv` | 1194 | point answers of Qwen2.5-VL-3B on the replication list: written point, correctness and per-item scores (112 errors) |
| `scores/points_replication_Qwen2.5-VL-7B.csv` | 1140 | the same for Qwen2.5-VL-7B (60 errors) |
| `scores/points_replication_Qwen3-VL-4B.csv` | 1187 | the same for Qwen3-VL-4B (84 errors) |
| `scores/points_heldout_Qwen2.5-VL-3B.csv` | 1195 | point answers of Qwen2.5-VL-3B on the held-out list (101 errors) |
| `scores/points_heldout_Qwen2.5-VL-7B.csv` | 1145 | the same for Qwen2.5-VL-7B (64 errors) |
| `scores/points_heldout_Qwen3-VL-4B.csv` | 1182 | the same for Qwen3-VL-4B (60 errors) |
| `standardization_points.csv` | 15 | per model and point-answer score: mean and (population) standard deviation of its out-of-fold values on the training items |
| `refits/point_gain_refits.csv` | 60 | point answers: gain in AUROC when 0.5 z(residual) is added to each score, for both lists together and for each list, by model and as the mean over the models: stored fit and the ten refits on training items redrawn with replacement |
| `results/boxes_<set>.csv` | | new items, RefCOCO+, held-out list, replication list and Objects365 (read-out fitted on RefCOCO): per model and item the IoU of the written box, the written box, the read-out box, the annotated box and the box written for the shifted view mapped back, in relative image coordinates (x1, y1, x2, y2 as fractions of the image width and height); `order` is the position of the item in the processing order of the read-out displacement analyses |
| `results/mean_training_box.csv` | | per model, the mean written box over the training items |
| `results/coco_instances.json` | | MS-COCO instance annotations (id, category, box, crowd flag) and image size of the images of these sets, from the MS-COCO 2014 annotations (CC BY 4.0) |
| `results/development_Qwen2.5-VL-3B.csv` | | Qwen2.5-VL-3B on four development sets (RefCOCO testA, testB and validation, RefCOCOg validation): COCO image and annotation id, IoU, two-teacher probe, residual, shift-teacher and sampling-teacher probe |
| `results/bicr_replication.csv` | | per model and item of the replication list, the score of our reimplementation of BICR |
| `results/bicr_heldout.csv` | 7197 | per model and item of the held-out list, the score of the same BICR fit: `model`, `item_id`, `bicr` |
| `results/bicr_standardization.csv` | 6 | per model, number of training items and the mean and (population) standard deviation of the out-of-fold BICR values on them; z(BICR) for BICR + residual uses these constants |
| `results/readout_error.csv` | 11 | read-out error sigma per model and fit (`fit`: RefCOCO, or Objects365 or VRSBench fitted in domain): `training_items` with a written and a read-out box, and `sigma`, the median distance between the centres of the out-of-fold read-out box and the written box over these items, in coordinates divided by the image width and height |
| `results/expected_main.csv` | | the numbers that `code/reproduce_main.py check` recomputes: place in the paper, quantity, value, decimals |
| `code/reproduce_main.py` | | script (Python 3 with numpy) for the numbers of the main-text tables and figures that `code/evaluate.py` does not print |
| `scores/matched-sizes/replication_Qwen2.5-VL-3B.csv` | 1200 | Qwen2.5-VL-3B on the replication list: IoU, correctness and, for training sizes 2,000, 5,000, 10,000, SPRED and the supervised linear probe on correctness (176 errors) |
| `scores/matched-sizes/heldout_Qwen2.5-VL-3B.csv` | 1200 | Qwen2.5-VL-3B on the held-out list: IoU, correctness and, for training sizes 2,000, 5,000, 10,000, SPRED and the supervised linear probe on correctness (178 errors) |
| `scores/matched-sizes/replication_Qwen2.5-VL-7B.csv` | 1199 | Qwen2.5-VL-7B on the replication list: IoU, correctness and, for training sizes 2,000, 5,000, 9,997, SPRED and the supervised linear probe on correctness (152 errors) |
| `scores/matched-sizes/heldout_Qwen2.5-VL-7B.csv` | 1199 | Qwen2.5-VL-7B on the held-out list: IoU, correctness and, for training sizes 2,000, 5,000, 9,997, SPRED and the supervised linear probe on correctness (152 errors) |
| `code/matched_sizes.py` | | script (Python 3 with numpy): AUROC of SPRED and of the probe from `scores/matched-sizes/` |
| `code/expected_matched.csv` | 38 | the numbers of the paper that `matched_sizes.py check` recomputes: place in the paper, quantity, value, decimals |
| `LICENSE` | | Creative Commons Attribution 4.0 International (CC BY 4.0), for the files of this package |
| `SHA256SUMS` | | SHA-256 of every other file in the package |

Row counts exclude the header line. All CSV files are UTF-8, comma-separated, with a header row; fields that contain commas or quotes are quoted as in RFC 4180. An empty field means the value is not defined for that item.

## Item lists (`lists/*.csv`)

| Column | Meaning |
|---|---|
| `item_id` | identifier of the item in this package (list code and position in the list); used by all other files |
| `dataset`, `split` | source dataset and split; RefCOCOg uses the UMD partition |
| `coco_image_id` | MS-COCO image id |
| `coco_annotation_id` | MS-COCO annotation id of the referred object |
| `expression` | referring expression given to the model |
| `box_x1`, `box_y1`, `box_x2`, `box_y2` | annotated box, pixels of the original image, origin at the top-left corner, x to the right, y downward; (x1, y1) top-left and (x2, y2) bottom-right corner, i.e. COCO [x, y, w, h] converted to [x, y, x + w, y + h] |
| `image_width`, `image_height` | size of the original image in pixels |

The lists of the two other image domains, `lists/objects365_test.csv` and `lists/vrsbench_test.csv`, have the columns `item_id`, `source_id`, `image_id`, the annotated box in pixels of the original image (as above) and the image size. `source_id` is the annotation id in the Ref-L4 tables (Objects365 part; `image_id` as in Ref-L4) or the `question_id` of `VRSBench_EVAL_referring.json` (`image_id` is the image file name without extension); the expressions are those of these sources. Both lists were drawn by us (one expression per box or per image, whole images on one side, fixed seed) and are not splits of the source datasets; a score file has fewer rows than its list where the model wrote no parseable box. In their score files the residual and the probes are standardized with the mean and (population) standard deviation of the training items of the same fit, as in the evaluation of these sets.

An object can carry several expressions, so an annotation id can occur in more than one item. In the RefCOCOg validation part of the new items two items occur twice with the same expression; both copies were scored, as in the paper.

Sources. Training items: RefCOCO validation split; new items: RefCOCO testB and RefCOCOg validation split (Hugging Face datasets `lmms-lab-encoder/RefCOCO`, revision `a5dff0b31947`, and `lmms-lab-encoder/RefCOCOg`, revision `93af55545967`); RefCOCO+ items: Hugging Face dataset `lmms-lab/RefCOCOplus`, revision `a283082593c4`; held-out list and replication list: RefCOCOg test split (`lmms-lab-encoder/RefCOCOg`, revision `93af55545967`). Training images are disjoint from the images of every test list.

`lists/training_extension.csv` holds the items added to the training items for the training sizes up to 10,000 (`scores/matched-sizes/`): referred objects of the RefCOCO train split on the original MS-COCO train2014 images. It has the columns of the other RefCOCO-family lists, except that `refcoco_ref_id` (ref id of the RefCOCO annotation) takes the place of `coco_annotation_id`; the box is the annotated box of that ref in pixels of the original image, as above. Training size n uses the first n items of `lists/training.csv` followed by this list, in this order; for Qwen2.5-VL-7B the 3 items of this list without a parsed answer are left out. Its images share no COCO image id with `lists/training.csv` or with any test list (`lists/new-items_refcoco-testB.csv` 0, `lists/new-items_refcocog-val.csv` 0, `lists/refcocoplus_testA.csv` 0, `lists/refcocoplus_val.csv` 0, `lists/heldout.csv` 0, `lists/replication.csv` 0 shared images); the Objects365 and VRSBench lists use other images.

### Checksums of the item lists

| List | SHA-256 of the CSV in this package |
|---|---|
| `lists/training.csv` | `3f0192dbde93d41f665084df63ca89d11af9b2437422598f515c800fd6ccf327` |
| `lists/new-items_refcoco-testB.csv` | `cbe032373f259f81a1dd04653b43ba1492da6d9a662aaa45c926917734c17d95` |
| `lists/new-items_refcocog-val.csv` | `4a40a5e1fcdd72a78719b503bd5bc97136ee6f31f2747b8e6bd4c65c063dba8a` |
| `lists/refcocoplus_testA.csv` | `c46a1995f990ea2f70be06b43a1f9b0fb70441a768133b8b248d644fc3d5b004` |
| `lists/refcocoplus_val.csv` | `c898e4ba0424d3320cb30d2a6df1ae8bff563dc1adf4571de424baff3c7aa04d` |
| `lists/heldout.csv` | `5e037693ce3a4e53352cf10a31806f054140d377f817c7350a66aa8cdf60d513` |
| `lists/replication.csv` | `43e2b9b621c511461e70588886b3f2fe7220b0a2f58eb552a6b2f6f5791b8db0` |
| `lists/training_extension.csv` | `9381399894eeeaa55802c011f6e1bd1c189b95ea3fd75225e37e1d2a41c0c4e6` |
| `lists/objects365_test.csv` | `e72d896cbcce970ad4f833284097187c7eb517a858af4b3297a193d5a47d22eb` |
| `lists/vrsbench_test.csv` | `2fc606e8e374b4fb26989d905f2470ad5d0cbbc2a86a58368f32ff73bac785ef` |

The held-out list is also given in JSON-lines form, `lists/heldout_refcocog_test.jsonl`, and so is the replication list, `lists/replication_refcocog_test.jsonl`; their checksums are in `SHA256SUMS`. Each line holds `id` (list name and COCO annotation id), `image_id`, `img` (local image file, `work/images/<coco_image_id>.jpg` as written by `python -m spred.items ... --image-dir work/images` of the code repository), `ref` (expression), `box` (annotated box as above), `W` and `H` (image size). The CSV of the same list has the same items in the same order.

## Written boxes and correctness (`answers/*.csv` and the first columns of `scores/*.csv`)

| Column | Meaning |
|---|---|
| `item_id` | item, see the item lists |
| `written_x1` ... `written_y2` | box written by the model (greedy decoding, official grounding prompt), mapped to pixels of the original image, same convention as the annotated box. For LLaVA-1.5-7B the image was padded to a square and the box was read on the padded square and mapped back |
| `iou` | IoU of the written box with the annotated box |
| `correct` | 1 if IoU >= 0.5, else 0 |
| `error_type` | `correct` (IoU >= 0.5), `wrong_object` (IoU < 0.1), `partial_overlap` (0.1 <= IoU < 0.3), `wrong_extent` (0.3 <= IoU < 0.5) |
| `readout_x1` ... `readout_y2` | `scores/*.csv` only: box read out by the fitted read-out from the hidden state of the last prompt token, pixels of the original image, same convention as the written box; `self_prediction_residual` is 1 - IoU of this box with the written box |
| `box_geometry_classifier` | `scores/*.csv` of the replication list, the held-out list and the new items, after the layer scores: logistic regression on eleven features of the written box (area, its square root, width, height, centre, distances of the centre from the image centre, log aspect ratio and its absolute value; relative image coordinates), fitted on the wrong answers of the training items with wrong-object errors as the positive class; decision value, larger = more likely a wrong-object error. Meant for the wrong answers only |
| `box_geometry_plus_residual_classifier` | the same classifier with the residual as a twelfth feature |
| `seven_view_disagreement` | new items, the four Qwen models, after the geometry columns: mean over seven further views of the item (four paddings, JPEG compression, noise, one sampled answer) of 1 - IoU between the box written for that view and the box written for the unchanged image; eight passes in all |
| `external_top_box_disagreement`, `external_support` | new items, Qwen2.5-VL-3B, Qwen2.5-VL-7B and Qwen3-VL-4B, last two columns: scores from a frozen open-vocabulary detector (IDEA-Research/grounding-dino-tiny, zero-shot) given the same expression; 1 - IoU between its highest-scoring box and the written box, and minus the largest product of its box confidence and IoU with the written box; larger = more suspicious; empty where the detector gave no output for the item |

The read-out columns are in every score file of box answers.

Only items with a parsed answer are listed; items whose answer could not be parsed into a box are absent from the file of that model. Boxes are rounded to 0.01 pixel and IoU to six decimals; `correct` and `error_type` are computed from the unrounded IoU.

Training items: written boxes only. The probe scores on the training items are in-sample or out-of-fold values of the fit itself and are not part of the evaluation.

## Per-item scores (`scores/*.csv`)

Every score is given in the direction used in the paper: larger means flagged as more likely wrong; AUROC is computed with errors as positives. All probes, read-outs and reference scores are the single fit on the 2,000 training items of each model reported in the paper; nothing was fitted on the test items. All scores use the generation pass except the content-shift check.

| Column | Meaning |
|---|---|
| `detector` | the proposed detector: z(two-teacher probe) + 0.5 z(self-prediction residual), each z standardized with the mean and standard deviation of its out-of-fold values on the training items (`standardization.csv`) |
| `two_teacher_probe` | label-free ridge probe on the hidden state, distilled from the shift and the sampling teacher |
| `self_prediction_residual` | self-prediction residual: 1 - IoU between the written box and the box read out by the fitted read-out from the hidden state of the last prompt token |
| `sampling_teacher_probe_plus_residual` | z(sampling-teacher probe) + 0.5 z(residual), standardized as for the detector |
| `linear_probe_correctness_plus_residual` | z(linear probe on correctness) + 0.5 z(residual), standardized as for the detector |
| `shift_teacher_probe` | label-free probe distilled from the shift teacher alone |
| `sampling_teacher_probe` | label-free probe distilled from the sampling teacher alone (semantic-entropy-probe style) |
| `mahalanobis` | Mahalanobis distance of the hidden state to the training items |
| `box_geometry_teacher` | ridge on the geometry of the written box fitted to the teacher |
| `coordinate_logprob` | minus the mean log-probability of the coordinate tokens (so that larger = more suspicious) |
| `in_box_attention` | attention of the generated digit tokens of the box to the image tokens whose centres lie inside the written box, averaged over all layers and heads (the in-box attention of the paper; a variant of MTLA); Qwen models only |
| `whole_image_attention` | the same attention summed over all image tokens, averaged over all layers and heads; Qwen models only |
| `linear_probe_correctness` | supervised linear probe on correctness (gold boxes of the training items) |
| `linear_probe_one_minus_iou` | supervised linear probe on 1 - IoU |
| `saplma` | supervised three-layer perceptron (SAPLMA) |
| `pregeneration_probe` | supervised probe on the hidden state before generation. For Qwen2.5-VL-7B cross-validation selected the same features, layer and penalty as for the correctness probe, so the two columns coincide |
| `linear_probe_wrong_object_all` | supervised linear probe on wrong-object labels, all training items |
| `linear_probe_wrong_object_vs_correct` | supervised linear probe on wrong-object labels, wrong-object and correct training items only |
| `box_geometry_supervised` | supervised ridge on box geometry |
| `content_shift_check` | content-shift check: 1 - IoU between the box written for the image and the box written for the image padded on the left by a strip in its mean colour (two patch-merging units for the Qwen models, 64 pixels for the other two), shifted back (two passes) |
| `chain_of_embedding_r`, `chain_of_embedding_c` | Chain-of-Embedding scores (CoE-R, CoE-C) of the answer tokens over all layers |
| `llm_check_hidden`, `llm_check_attention` | LLM-Check hidden-state score and attention score at the middle layer (half the number of layers, rounded down); the attention score for the Qwen models only |
| `logit_lens_box` | 1 - IoU between the written box and the box decoded by the model's own output head from the layer at three quarters of the depth |
| `logit_lens_logprob` | minus the mean log-probability of the written coordinate tokens under the output head at that layer |
| `*_best_layer` | the same score at the layer that gives the highest AUROC on these items (the layer is chosen with the labels of these items) |

The training-free layer scores (the columns after `content_shift_check`, before the geometry columns) are given for the new items and the replication list. For them the sign was set per score as the better of the two directions on the items scored.

Scores that are not defined for a model are omitted: the two attention scores for InternVL2.5-8B-MPO and LLaVA-1.5-7B.

## Evaluation code (`code/`)

`code/evaluate.py` needs Python 3 and numpy and nothing else: no model, no image, no hidden state. Run it from the top directory of the package (or pass `--root`):

```
python code/evaluate.py check                                   # every number below against the paper (about a minute)
python code/evaluate.py counts --set replication
python code/evaluate.py table4 --set replication
python code/evaluate.py gain   --set replication --errors wrong_object
python code/evaluate.py gain   --set replication --errors all
python code/evaluate.py review --set replication --share 0.2
python code/evaluate.py types  --set replication
python code/evaluate.py pull   --set replication
python code/evaluate.py pairs
python code/evaluate.py pairs  --set replication
python code/evaluate.py views
python code/evaluate.py points
python code/evaluate.py others
```

`--set` takes `replication`, `heldout` or `new-items` (default `replication`; for `pairs` the new items); `--errors` takes `all`, `wrong_object`, `partial_overlap` or `wrong_extent`. `check` prints, for each of the 582 numbers in `code/expected.csv`, where it stands in the paper, the recomputed value and the value in the paper, and exits with an error if any differs after rounding. The intervals are reproduced exactly because the resampling is fixed: seed 0; for means over models 2,000 resamples of the images of all models together, images ordered by their id as text; for the per-model bound 10,000 resamples of the images of that model.

### Point answers (`scores/points_*.csv`)

The model is asked for one point on the referred object instead of a box (Qwen2.5-VL-3B: `Locate "{ref}" in the image, and output its point coordinates in JSON format.`; Qwen2.5-VL-7B and Qwen3-VL-4B: `Point to "{ref}" in the image, and output the point in JSON format with the key "point_2d".`). The items are those of `lists/replication.csv` and `lists/heldout.csv`; an item whose answer could not be parsed into a point is absent from the file of that model. The probes and the read-out are fitted on about 1,000 unlabeled training items with point answers; nothing is fitted on the two lists.

| Column | Meaning |
|---|---|
| `item_id` | item, see the item lists |
| `point_x`, `point_y` | point written by the model (greedy decoding), pixels of the original image |
| `correct` | 1 if the point lies inside the annotated box, else 0 |
| `two_teacher_probe` | label-free ridge probe distilled from the shift and the sampling teacher (distance between the greedy point and the point of the shifted view or of one sampled answer) |
| `self_prediction_residual` | distance, in relative image units, between the written point and the point read out by the fitted read-out from the hidden state of the last prompt token |
| `sampling_teacher_probe` | label-free probe distilled from the sampling teacher alone |
| `linear_probe_correctness` | supervised linear probe on correctness (labels of the training items) |
| `saplma` | supervised three-layer perceptron on the same feature |
| `coordinate_logprob` | minus the mean log-probability of the coordinate tokens |

Larger means flagged as more likely wrong. z(score) uses `standardization_points.csv`; the perceptron has no out-of-fold training values and is standardized with its mean and standard deviation on the items that are scored.

## Further main-text numbers (`code/reproduce_main.py`)

`code/reproduce_main.py` recomputes, from the files above and `results/`, the numbers of the main-text tables and figures that `code/evaluate.py` does not print: the items per list (Table 2), the BICR rows and the per-model review rows of Table 4, the BICR rows of Table S9.2 (held-out list), the ablation (Table 5), the second answer against the read-out (Figure 3), the displacement of the read-out (Figure 4), the counts of models within 0.03 of the supervised probe (Section 5.2), the read-out error with the share of written boxes narrower than twice it (Section 3.6), the wrong-object errors on another annotated instance (Introduction and Section 6.3), the accuracy of the read-out box (Section 3.3), and SPRED on the two other image domains with the gain by target width (Sections 7 and 3.6). It needs Python 3 and numpy and imports `code/evaluate.py`:

```
python code/reproduce_main.py check
python code/reproduce_main.py lists
python code/reproduce_main.py table4
python code/reproduce_main.py ablation
python code/reproduce_main.py second-answer
python code/reproduce_main.py pull
python code/reproduce_main.py margin
python code/reproduce_main.py narrow
python code/reproduce_main.py instances
python code/reproduce_main.py readout-accuracy
python code/reproduce_main.py domains
```

`check` compares each of the 444 numbers in `results/expected_main.csv` with the recomputed value and exits with an error if any differs after rounding. The resampling is fixed (seed 0); the order of the items and of the models follows the files.

BICR on the held-out list (`table4`): AUROC of `bicr` in `results/bicr_heldout.csv` against 1 - `correct` of `scores/heldout_<model>.csv` (all errors) and for wrong-object errors against correct answers; BICR + residual is z(BICR) + 0.5 z(residual), z(BICR) with `results/bicr_standardization.csv` and z(residual) with `standardization.csv`.

Target size (`narrow`): the width of a written box is the square root of its area in coordinates divided by the image width and height (written box of `scores/<set>_<model>.csv`, image size of the item list); it is compared with twice `sigma` of `results/readout_error.csv` for the fit that produced the scores: RefCOCO for the held-out list and the new items, the in-domain fit for `scores/objects365_fitted-in-domain_<model>.csv` and `scores/vrsbench_fitted-in-domain_<model>.csv`.

Instances (`instances`): a wrong-object error of the replication list lies on another annotated instance when its written box has IoU >= 0.5 with a non-crowd MS-COCO instance of `results/coco_instances.json` other than the target instance (the instance with the highest IoU to the annotated box, which must reach 0.5); boxes of `results/boxes_replication.csv`. Read-out accuracy (`readout-accuracy`): share of the read-out boxes and of the written boxes with IoU >= 0.5 with the annotated box. Other image domains (`domains`): AUROC of `detector` in `scores/<domain>_fitted-in-domain_<model>.csv`; the gain by target width uses `scores/objects365_fitted-on-RefCOCO_<model>.csv`, `standardization.csv` and the RefCOCO `sigma` of `results/readout_error.csv`, as for `narrow`.

## Matched training sizes (`scores/matched-sizes/`, `code/matched_sizes.py`)

For Qwen2.5-VL-3B and Qwen2.5-VL-7B the training items were extended with referred objects of the RefCOCO train split (`lists/training_extension.csv`) to 10,000 answered items (9,997 for Qwen2.5-VL-7B). Column `spred_<n>`: SPRED fitted on the first n training items without their labels (weight 0.5, standardized with the constants of that fit); column `probe_<n>`: the supervised linear probe on correctness trained on the same n items with their labels, its feature, layer and penalty searched at that size. The rows follow the item lists; `iou`, `correct` and `error_type` are those of the other files.

```
python code/matched_sizes.py table
python code/matched_sizes.py check
```

`check` compares each of the 38 numbers in `code/expected_matched.csv` with the recomputed value and exits with an error if any differs after rounding. The intervals of these comparisons add ten refits on redrawn training items, which are not part of this package.

## Recomputing the reported numbers

All-error AUROC: items with `correct` = 0 against all other items. Error-type AUROC: items of that `error_type` against the items with `correct` = 1. Intervals in the paper come from a bootstrap over images (all items of a sampled image enter together; `coco_image_id` from the item lists). For example, the detector AUROC for all errors on the held-out list of a model is the AUROC of `detector` against 1 - `correct` in `scores/heldout_<model>.csv`.

Standardization: z(score) = (value - training_mean) / training_sd with the constants of `standardization.csv`. Before standardizing, a missing value of a score on test items is set to the largest value of that score in the same file; the constants themselves use the non-missing training values only (the probe on wrong-object and correct items is missing on the training items it was not trained on). With this rule the gains of Table 3 and Figure 2 follow from the per-item scores. Their combined intervals add the spread over ten refits on redrawn training items: the stored gain plus or minus 1.96 sqrt(SE^2 + SD^2), with SE from the bootstrap over images on the per-item scores and SD the standard deviation of the ten refit gains in `refits/plugin_gain_refits.csv`.

Which file, columns and command give each number:

| Place in the paper | Number | Files and columns | Command |
|---|---|---|---|
| Abstract | items of the replication list | `lists/replication.csv` | `counts --set replication` |
| Highlights | share of wrong-object errors caught with the top fifth reviewed, 46.9% to 57.5%, mean over fifteen scores and the models | `scores/replication_<model>.csv`: the fifteen score columns, `self_prediction_residual`, `error_type`; `standardization.csv` | `review --set replication --share 0.2` (last line) |
| Contribution 1; Section 5.1 | 0.726: among wrong answers, the residual tells wrong-object errors from the other errors | `scores/replication_<model>.csv`: `self_prediction_residual`, `error_type` | `types --set replication` |
| Section 5.2; abstract (bound on the interval) | all errors, detector against its probe: +0.007 with its combined interval | `scores/replication_<model>.csv`: `two_teacher_probe`, `self_prediction_residual`, `error_type`; `lists/replication.csv`: `coco_image_id`; `standardization.csv`; `refits/plugin_gain_refits.csv` | `gain --set replication --errors all` (row `two_teacher_probe`) |
| Contribution 1; Section 5.1 (interval) | 0.061 with its combined interval: gain when the residual is added to the classifier on box geometry, among wrong answers | `scores/replication_<model>.csv`: `box_geometry_classifier`, `box_geometry_plus_residual_classifier`, `error_type`; `refits/geometry_classifier_gain_refits.csv` | `types --set replication` (last line) |
| Introduction; Section 5.1 | 0.607: wrong-object errors caught by the detector (against 0.505 for the probe alone in Section 5.1) | as above, plus `coordinate_logprob`, `linear_probe_correctness` | `review --set replication --share 0.2` |
| Abstract; introduction; Sections 3.6 and 5.1; Discussion | 0.518 to 0.524: share of all errors caught by the probe alone and by the detector with the top fifth reviewed; 0.86: cost of a wrong-object error, in units of another error, at which the detector breaks even with its probe in cost-weighted recall; +0.006: detector minus probe in cost-weighted recall at equal costs | `scores/replication_<model>.csv`: `two_teacher_probe`, `self_prediction_residual`, `error_type`; `standardization.csv` | `review --set replication --share 0.2` (last line) |
| Discussion; supplementary table of the external detector | a frozen open-vocabulary detector as second localizer: AUROC of its two scores on wrong-object errors, and the gains when they are added to the two-teacher probe with and without the residual (weights 0.5) | `scores/new-items_<model>.csv` (Qwen2.5-VL-3B, Qwen2.5-VL-7B, Qwen3-VL-4B): `external_top_box_disagreement`, `external_support`, `two_teacher_probe`, `self_prediction_residual`, `error_type`; `standardization.csv` for the probe and the residual; the external scores are standardized with their mean and standard deviation on the scored items | from the columns (no command; the point estimates follow, the refit part of the intervals does not) |
| Discussion | 0.078 and 0.070: lead of the detector over the disagreement across seven views, new items, the two Qwen3-VL models | `scores/new-items_<model>.csv`: `two_teacher_probe`, `self_prediction_residual`, `seven_view_disagreement`, `error_type`; `standardization.csv` | `views` |
| Section 6.3 | 0.250: the read-out lies partway towards the annotated object | `scores/replication_<model>.csv`: `readout_*`, `written_*`, `error_type`; `lists/replication.csv`: box and image size; `answers/training_<model>.csv` and `lists/training.csv` (mean written box) | `pull --set replication` |
| Discussion | 0.599: on items that one model answers correctly and another wrongly, the residual picks the model that erred | `scores/new-items_<model>.csv` (five models): `self_prediction_residual`, `error_type` | `pairs` |
| Discussion | 0.637 with its combined interval: the same, counting only pairs in which the erring model placed its box on another object | `scores/new-items_<model>.csv` (five models): `self_prediction_residual`, `iou`; `lists/new-items_*.csv`: `coco_image_id`, `coco_annotation_id`; `refits/pair_test_refits.csv` | `pairs` |
| Section 6.1; supplementary tables of the own and the other models' residuals | gain with the own residual, with the mean residual of the other five models on the same item, their ratio with its interval over images; per model also with one other model's residual | `scores/replication_<model>.csv`, `scores/heldout_<model>.csv`: the fifteen score columns, `self_prediction_residual`, `error_type`; `standardization.csv`; `standardization_others_residual.csv`; `refits/others_residual_gain_refits.csv` (only for the count of scores whose combined interval is above zero) | `others` |
| Abstract; contribution 3; Section 5.1 | 0.026 to 0.049: gain on wrong-object errors for the probes trained on 2,000 labelled items (smallest: linear probe on wrong-object labels; largest: SAPLMA perceptron) | as for Table 3 | `gain --set replication --errors wrong_object` |
| Introduction, scope | 27.8% to 48.3%: share of wrong-object errors among the errors | `scores/replication_<model>.csv`: `error_type` | `counts --set replication` |
| Table 3 | gain on wrong-object errors for each of the fifteen scores, replication list; bold when the combined interval excludes zero | fifteen score columns, `self_prediction_residual`, `error_type`; `standardization.csv`; `refits/plugin_gain_refits.csv` | `gain --set replication --errors wrong_object` |
| Table 4 | items, errors, AUROC rows per model, bound against the supervised probe (BICR rows and the review rows: `code/reproduce_main.py table4`) | `scores/replication_<model>.csv`: `two_teacher_probe`, `self_prediction_residual`, `coordinate_logprob`, `linear_probe_correctness`, `error_type`; `lists/replication.csv`: `coco_image_id`; `standardization.csv`; `results/bicr_replication.csv` | `counts --set replication`, `table4 --set replication` |
| Section 5.2; Table S4.3 | SPRED against the supervised linear probe on correctness with matched training sizes up to 10,000 items, two models: AUROC per size and list, SPRED minus the probe and its mean over the models at the largest size | `scores/matched-sizes/<list>_<model>.csv`: `spred_<n>`, `probe_<n>`, `correct` | `python code/matched_sizes.py table` (point estimates; the intervals need refits not included) |
| Table S9.2 | BICR rows of the held-out list: AUROC for all errors and for wrong-object errors, detector minus BICR, BICR + residual minus BICR on wrong-object errors | `results/bicr_heldout.csv`, `results/bicr_standardization.csv`; `scores/heldout_<model>.csv`: `two_teacher_probe`, `self_prediction_residual`, `iou`; `standardization.csv` | `python code/reproduce_main.py table4` |
| Section 3.6; Discussion; Section S7 | read-out error sigma per fit, and the share of written boxes narrower than 2 sigma: at most 0.015 of the items of any model on the RefCOCO family, 0.599 to 0.665 on Objects365 and 0.543 to 0.590 on remote-sensing images (read-out fitted on the domain) | `results/readout_error.csv`; `scores/heldout_<model>.csv`, `scores/new-items_<model>.csv`, `scores/objects365_fitted-in-domain_<model>.csv`, `scores/vrsbench_fitted-in-domain_<model>.csv`: `written_*`; the item lists: image size | `python code/reproduce_main.py narrow` |
| Figure 2 | per-model gains (markers), their mean and its combined interval (tick and line), replication and held-out list | as for Table 3, with `scores/heldout_<model>.csv` for the held-out panel | `gain --set replication --errors wrong_object`, `gain --set heldout --errors wrong_object` |
| Contribution 3; Section 5.1, point answers | +0.035 with its combined interval: gain over all errors when the residual is added to the two-teacher probe, both lists, mean over three models | `scores/points_replication_<model>.csv`, `scores/points_heldout_<model>.csv`: `two_teacher_probe`, `self_prediction_residual`, `correct`; `lists/replication.csv`, `lists/heldout.csv`: `coco_image_id`; `standardization_points.csv`; `refits/point_gain_refits.csv` | `points` (first block; the other rows and blocks give the other scores and each list) |
| Section 5.1; Table S6.8 | among wrong answers, AUROC of wrong-object errors against the other errors for each row of the table, mean over the models with its interval over images; the residual added to the classifier on box geometry | `scores/replication_<model>.csv`: the score columns, `box_geometry_classifier`, `box_geometry_plus_residual_classifier`, `written_*`, `error_type`; `lists/replication.csv`: image size, `coco_image_id`; `standardization.csv`; `refits/geometry_classifier_gain_refits.csv` | `types --set replication` |
| Introduction (contributions); Section 7 | 0.608 [0.543, 0.674]: the same pair test on the replication list, wrong-object pairs, with its interval over images | `scores/replication_<model>.csv` (five models): `self_prediction_residual`, `iou`; `lists/replication.csv`: `coco_image_id`, `coco_annotation_id` | `pairs --set replication` |
| Introduction; Section 6.3 | 72.6%: wrong-object errors of the replication list whose written box lies on another annotated instance; SPRED minus its probe on them, 0.048 | `results/boxes_replication.csv`, `results/coco_instances.json`; `scores/replication_<model>.csv`: `two_teacher_probe`, `self_prediction_residual`, `iou`; `standardization.csv` | `python code/reproduce_main.py instances` |
| Section 3.3 | acc@0.5 of the read-out box, 0.391 to 0.467, and of the written box, 0.853 to 0.876 (replication list, first three models) | `scores/replication_<model>.csv`: `readout_*`, `written_*`; `lists/replication.csv`: annotated box | `python code/reproduce_main.py readout-accuracy` |
| Section 7; Section 3.6 | AUROC of SPRED on Objects365, 0.692 to 0.762, and on VRSBench, 0.752 and 0.784; with the read-out fitted on RefCOCO, the gain on wrong-object errors for written boxes narrower and wider than 2 sigma, 0.013 and 0.005 | `scores/objects365_*_<model>.csv`, `scores/vrsbench_fitted-in-domain_<model>.csv`; `lists/objects365_test.csv`; `standardization.csv`; `results/readout_error.csv` | `python code/reproduce_main.py domains` |

Not recomputable from this package, because they need hidden states, further model passes or fits on the training items:

- the comparisons with the read-out of another item, the read-out fitted to annotated boxes, and every baseline that reads hidden states or attention at other layers than the exported scores;
- the fitting itself: the two classifiers on box geometry and the seven-view answers enter as per-item columns, not as code that refits them;
- combined intervals other than those of the gains and of the pair test: the files under `refits/` hold the refits of these only; the image part of every other interval follows from the per-item scores;
- for the training sizes up to 10,000 items, every score other than SPRED and the linear probe on correctness, and the fits on blocks of the training items;
- numbers that need material outside this package: the share of errors whose box all seven perturbed views keep and the AUROC of SPRED on these items (Introduction, Section 7; the package holds the mean disagreement over the views, not the box of each view), the share of wrong-object errors on annotations that Ref-L4 found faulty (Section 6.1), the run time of SPRED relative to generation alone (Section 3.6), and the fits on subsets of 250 labelled training items and with one error type left out of the training items (Section 5.3).

Notes for `others` and `pairs`. The mean residual of the other models for an item is the mean of their standardized residuals over the models that answered it, standardized again with `standardization_others_residual.csv`; one other model's residual enters with its own standardization, and 0 where that model has no answer. In the pair test an item that is listed twice in `lists/new-items_refcocog-val.csv` (same annotation and expression) is counted once; on the new items its intervals combine the images and the refits of `refits/pair_test_refits.csv`, on another item set (`--set`) they are over images only.

Model checkpoints (Hugging Face): Qwen2.5-VL-3B: `Qwen/Qwen2.5-VL-3B-Instruct`; Qwen2.5-VL-7B: `Qwen/Qwen2.5-VL-7B-Instruct`; Qwen3-VL-4B: `Qwen/Qwen3-VL-4B-Instruct`; Qwen3-VL-8B: `Qwen/Qwen3-VL-8B-Instruct`; InternVL2.5-8B-MPO: `OpenGVLab/InternVL2_5-8B-MPO-hf`; LLaVA-1.5-7B: `llava-hf/llava-1.5-7b-hf`. Revisions are listed in Section S1 of the supplementary material.
