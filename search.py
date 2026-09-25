# Text -> image search with the trained model (used by Flask_Web.py, also runnable from the command line)
import argparse
import json
import os
import re

import numpy as np
import torch

from rank import CrossModalNet

# ── Paths (update these to match your environment) ──────────────────────────
feature_folder = "./features/"
model_path     = "./models/net.pt"
index_path     = "./meta_data/output.json"
pic_folder     = "./data/Flicker8k_Dataset/"
# ─────────────────────────────────────────────────────────────────────────────


class Searcher:
    """Embeds every image once at start-up, then ranks them by cosine similarity to a text query."""

    def __init__(self, splits=("train", "val", "test")):
        checkpoint = torch.load(model_path)
        self.net = CrossModalNet(**checkpoint["config"])
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

        # Gallery = every Flickr8k image
        self.pic_ids, feats = [], []
        for split in splits:
            with open(os.path.join(feature_folder, f"img_{split}_ids.json"), encoding="utf8") as fp:
                self.pic_ids += json.load(fp)
            feats.append(np.load(os.path.join(feature_folder, f"img_{split}.npy")))
        with torch.no_grad():
            self.img_emb = self.net.encode_image(torch.from_numpy(np.concatenate(feats)))

        self.vectors = np.load(os.path.join(feature_folder, "glove_vectors.npy"))
        with open(os.path.join(feature_folder, "glove_words.json"), encoding="utf8") as fp:
            self.word_to_index = {w: i for i, w in enumerate(json.load(fp))}
        with open(index_path, encoding="utf8") as fp:
            self.inverted_index = json.load(fp)

    @staticmethod
    def tokenize(text):
        return re.findall(r"[a-z]+", text.lower())

    def semantic(self, query, top_k=12):
        """Rank all images by the model's image-text similarity. Returns [(pic_id, score)]."""
        rows = [self.word_to_index[w] for w in self.tokenize(query) if w in self.word_to_index]
        if not rows:
            return []
        with torch.no_grad():
            txt_emb = self.net.encode_text(torch.from_numpy(self.vectors[rows].mean(axis=0)))
        scores = self.img_emb @ txt_emb
        best = scores.topk(min(top_k, len(scores)))
        return [(self.pic_ids[i], s) for s, i in zip(best.values.tolist(), best.indices.tolist())]

    def keyword(self, query, top_k=12):
        """Baseline: images whose captions contain every query word (inverted index), unranked."""
        words = self.tokenize(query)
        if not words or any(w not in self.inverted_index for w in words):
            return []
        hits = set.intersection(*(set(self.inverted_index[w]) for w in words))
        return [(pic_id, None) for pic_id in sorted(hits)[:top_k]]


def save_grid(query, results, out_path, thumb=256):
    """Save the top results side by side with the query as a title (for the README)."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    header = 40
    grid = Image.new("RGB", (thumb * len(results), thumb + header), "white")
    ImageDraw.Draw(grid).text((10, 8), f'"{query}"', fill="black", font=font)
    for i, (pic_id, score) in enumerate(results):
        img = Image.open(os.path.join(pic_folder, pic_id + ".jpg")).convert("RGB")
        side = min(img.size)  # center crop to a square thumbnail
        left, top = (img.width - side) // 2, (img.height - side) // 2
        grid.paste(img.crop((left, top, left + side, top + side)).resize((thumb, thumb)), (i * thumb, header))
    grid.save(out_path, quality=85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Flickr8k images with a text query")
    parser.add_argument("query", nargs="+", help="one or more queries")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mode", choices=["semantic", "keyword"], default="semantic")
    parser.add_argument("--gallery", choices=["all", "test"], default="all",
                        help="test: only search the 1,000 test images the model never saw during training")
    parser.add_argument("--save_dir", default=None, help="also save a result strip per query as <save_dir>/<query>.jpg")
    args = parser.parse_args()

    searcher = Searcher(("test",) if args.gallery == "test" else ("train", "val", "test"))
    for query in args.query:
        results = getattr(searcher, args.mode)(query, args.top_k)
        print(f"\n{query}")
        for pic_id, score in results:
            print(f"  {pic_id}" + (f"  {score:.3f}" if score is not None else ""))
        if args.save_dir and results:
            os.makedirs(args.save_dir, exist_ok=True)
            save_grid(query, results, os.path.join(args.save_dir, "_".join(Searcher.tokenize(query)) + ".jpg"))
