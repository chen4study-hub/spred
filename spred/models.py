"""Loading, prompting and answer parsing for the three model families.

qwen      Qwen2.5-VL and Qwen3-VL: the image is resized with the processor's own rule to between 256 and 1024 merged
          tokens before it is shown; boxes and points in pixels (Qwen2.5-VL) or on a 0-1000 grid (Qwen3-VL); the
          shifted view pads two merged-token widths on the left. Point answers are asked of the Qwen models only.
internvl  InternVL2.5 (Hugging Face format): thumbnail plus up to six tiles; boxes on a 0-1000 grid; the shifted view
          pads 64 pixels on the left.
llava     LLaVA-1.5: the image is padded to a square in the processor's mean colour; boxes as fractions of that
          square; the shifted view pads 64 pixels on the left.
Boxes are returned in pixels of the image that was shown (for llava: of the unpadded image).
"""
from spred import geometry as G

QWEN_PROMPT = 'Locate "{ref}" in the image, and output its bbox coordinates in JSON format.'
POINT_PROMPTS = {  # point answers: Qwen2.5-VL-3B was asked with 'plain', Qwen2.5-VL-7B and Qwen3-VL-4B with 'keyed'
    'plain': 'Locate "{ref}" in the image, and output its point coordinates in JSON format.',
    'keyed': 'Point to "{ref}" in the image, and output the point in JSON format with the key "point_2d".',
}
INTERNVL_PROMPT = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{ref}</ref>"
LLAVA_PROMPT = "Please provide the bounding box coordinate of the region this sentence describes: {ref}."
MIN_PIXELS, MAX_PIXELS = 256 * 28 * 28, 1024 * 28 * 28


def family_of(repo):
    r = repo.lower()
    return "qwen" if "qwen" in r else "internvl" if "internvl" in r else "llava" if "llava" in r else None


def pad_left(img, pad):
    """The image with a strip of `pad` pixels in its mean colour added on the left."""
    from PIL import Image
    fill = tuple(int(v) for v in img.resize((1, 1)).getpixel((0, 0)))
    out = Image.new("RGB", (img.size[0] + pad, img.size[1]), fill)
    out.paste(img, (pad, 0))
    return out


class Runner:
    """Common interface: image(item) -> (shown image, annotated box in its pixels); inputs(image, ref); answer(inputs,
    max_new, seed); box(text, image) -> pixels or None; shift_pad; coord_after (marker before the box in the answer)."""
    max_new = 64
    shift_pad = 64
    coord_after = ""
    attention = False

    def answer(self, inp, max_new, seed=None, temperature=0.7, top_p=0.9):
        import torch
        kw = dict(max_new_tokens=max_new, do_sample=False)
        if seed is not None:
            torch.manual_seed(seed)
            kw = dict(max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p)
        out = self.model.generate(**inp, **kw)
        gen = out[0, inp["input_ids"].shape[1]:].tolist()
        return gen, self.decode(gen)

    def image(self, it):
        from PIL import Image
        img = Image.open(it["img"]).convert("RGB")
        W, H = img.size
        return img, G.scale_box(it["box"], W / it["W"], H / it["H"])


class QwenRunner(Runner):
    coord_after = "bbox_2d"
    attention = True

    def __init__(self, repo, attn="sdpa"):
        import torch
        import transformers
        cfg = transformers.AutoConfig.from_pretrained(repo)
        cls = {"qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration", "qwen3_vl": "Qwen3VLForConditionalGeneration"}[cfg.model_type]
        self.model = getattr(transformers, cls).from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn).eval()
        hi = int(MAX_PIXELS * 1.6)          # above the resize target, so that the processor keeps the size chosen here
        self.proc = transformers.AutoProcessor.from_pretrained(repo, min_pixels=MIN_PIXELS, max_pixels=hi)
        ip = self.proc.image_processor
        for k, v in (("min_pixels", MIN_PIXELS), ("max_pixels", hi)):
            if hasattr(ip, k):
                setattr(ip, k, v)
        if isinstance(getattr(ip, "size", None), dict) and "longest_edge" in ip.size:
            ip.size = {"shortest_edge": MIN_PIXELS, "longest_edge": hi}
        self.factor = ip.patch_size * ip.merge_size
        self.coord = "abs" if cfg.model_type == "qwen2_5_vl" else "norm1000"
        self.shift_pad = 2 * self.factor
        self.image_token_id = self.model.config.image_token_id
        self.n_layers = self.model.config.get_text_config().num_hidden_layers
        self.template = QWEN_PROMPT

    def image(self, it):
        from PIL import Image
        im0 = Image.open(it["img"]).convert("RGB")
        Hr, Wr = G.smart_resize(it["H"], it["W"], self.factor, MIN_PIXELS, MAX_PIXELS)
        return im0.resize((Wr, Hr), Image.BICUBIC), G.scale_box(it["box"], Wr / it["W"], Hr / it["H"])

    def inputs(self, img, ref):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.template.format(ref=ref)}]}]
        chat = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[chat], images=[img], return_tensors="pt").to("cuda")
        gh, gw = inp["image_grid_thw"][0, 1:].tolist()
        ps = self.proc.image_processor.patch_size
        assert (gw * ps, gh * ps) == img.size, f"the processor resized the image ({img.size} vs grid {gh}x{gw} x {ps})"
        return inp

    def decode(self, gen):
        return self.proc.tokenizer.decode(gen, skip_special_tokens=True)

    def box(self, text, img):
        return G.qwen_to_pixels(G.parse_box_qwen(text), self.coord, *img.size)

    def point(self, text, img):
        return G.qwen_point_to_pixels(G.parse_point(text), self.coord, *img.size)


class InternVLRunner(Runner):
    def __init__(self, repo, max_patches=6):
        import torch
        import transformers
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = transformers.AutoProcessor.from_pretrained(repo)
        ip = self.proc.image_processor
        ip.crop_to_patches = max_patches > 1
        ip.max_patches = max_patches
        self.n_layers = self.model.config.text_config.num_hidden_layers

    def inputs(self, img, ref):
        import torch
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INTERNVL_PROMPT.format(ref=ref)}]}]
        chat = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        return self.proc(text=[chat], images=[img], return_tensors="pt").to("cuda", dtype=torch.bfloat16)

    def decode(self, gen):
        return self.proc.tokenizer.decode(gen, skip_special_tokens=False).replace("<|im_end|>", "").strip()

    def box(self, text, img):
        return G.internvl_to_pixels(G.parse_box_internvl(text), *img.size)


class LlavaRunner(Runner):
    max_new = 48

    def __init__(self, repo):
        import torch
        import transformers
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.proc = transformers.AutoProcessor.from_pretrained(repo)
        self.n_layers = self.model.config.text_config.num_hidden_layers

    def inputs(self, img, ref):
        import torch
        from PIL import Image
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": LLAVA_PROMPT.format(ref=ref)}]}]
        chat = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        W, H = img.size
        S, ox, oy = G.square_offsets(W, H)
        sq = Image.new("RGB", (S, S), tuple(int(round(255 * m)) for m in self.proc.image_processor.image_mean))
        sq.paste(img, (int(ox), int(oy)))
        return self.proc(text=[chat], images=[sq], return_tensors="pt").to("cuda", dtype=torch.bfloat16)

    def decode(self, gen):
        return self.proc.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def box(self, text, img):
        return G.llava_to_pixels(G.parse_box_llava(text), *img.size)


def load(repo, attn="sdpa"):
    fam = family_of(repo)
    if fam == "qwen":
        return QwenRunner(repo, attn)
    if fam == "internvl":
        return InternVLRunner(repo)
    if fam == "llava":
        return LlavaRunner(repo)
    raise SystemExit(f"unsupported model: {repo}")
