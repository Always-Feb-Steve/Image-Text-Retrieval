# Build train/val/test metadata from the Flickr8k captions and official split
import json
import os
from collections import defaultdict

# ── Paths (update these to match your environment) ──────────────────────────
text_dir  = "./data/Flickr8k_text"
pic_folder = "./data/Flicker8k_Dataset"
out_dir   = "./meta_data"
# ─────────────────────────────────────────────────────────────────────────────

splits = {
    "train": "Flickr_8k.trainImages.txt",  # 6,000 images
    "val":   "Flickr_8k.devImages.txt",    # 1,000 images (model selection)
    "test":  "Flickr_8k.testImages.txt",   # 1,000 images (final metrics)
}

# Flickr8k.token.txt lines look like: "1000268201_693b08cb0e.jpg#0\tA child in a pink dress ..."
captions = defaultdict(list)
with open(os.path.join(text_dir, "Flickr8k.token.txt"), encoding="utf8") as fp:
    for line in fp:
        if "\t" not in line:
            continue
        key, caption = line.rstrip("\n").split("\t", 1)
        pic_id = key.split("#")[0].rsplit(".jpg", 1)[0]
        captions[pic_id].append(caption.strip())

os.makedirs(out_dir, exist_ok=True)
for split, list_file in splits.items():
    with open(os.path.join(text_dir, list_file), encoding="utf8") as fp:
        pic_ids = [line.strip().rsplit(".jpg", 1)[0] for line in fp if line.strip()]

    # Keep only images that exist on disk and have captions
    rows = [{"pic_id": p, "captions": captions[p]} for p in pic_ids
            if captions[p] and os.path.exists(os.path.join(pic_folder, p + ".jpg"))]

    # One JSON object per line, like the original input.json
    with open(os.path.join(out_dir, f"{split}_data.json"), "w", encoding="utf8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("#{} images: {}  captions: {}".format(split, len(rows), sum(len(r["captions"]) for r in rows)))
