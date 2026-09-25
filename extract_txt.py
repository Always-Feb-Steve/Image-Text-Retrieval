# Extract text features from captions using GloVe word embeddings (mean of word vectors)
import json
import os
import re
import time

import numpy as np

# ── Paths (update these to match your environment) ──────────────────────────
word_file      = "./data/glove.6B.200d.txt"
meta_folder    = "./meta_data/"
feature_folder = "./features/"
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 50000  # GloVe is sorted by frequency; the web demo only needs the common words

os.makedirs(feature_folder, exist_ok=True)


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def embed(text, word_to_index, vectors):
    """Average the vectors of the known words; None if no word is known."""
    rows = [word_to_index[w] for w in tokenize(text) if w in word_to_index]
    if not rows:
        return None
    return vectors[rows].mean(axis=0)


if __name__ == "__main__":
    # ── Build GloVe lookup ───────────────────────────────────────────────────
    print("---------------------Building GloVe dict---------------------------")
    t0 = time.perf_counter()
    words, vectors = [], []
    with open(word_file, "r", encoding="utf8") as fp:
        for line in fp:
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
    vectors = np.stack(vectors)  # (400000, 200)
    word_to_index = {w: i for i, w in enumerate(words)}
    print("GloVe dict size:", len(words), " build time: {:.1f}s".format(time.perf_counter() - t0))

    # Save a compact vocabulary for query-time use (Flask_Web.py / search.py)
    np.save(os.path.join(feature_folder, "glove_vectors.npy"), vectors[:VOCAB_SIZE])
    with open(os.path.join(feature_folder, "glove_words.json"), "w", encoding="utf8") as fp:
        json.dump(words[:VOCAB_SIZE], fp)

    # ── Embed every caption ──────────────────────────────────────────────────
    for split in ["train", "val", "test"]:
        with open(os.path.join(feature_folder, f"img_{split}_ids.json"), encoding="utf8") as fp:
            img_index = {pic_id: i for i, pic_id in enumerate(json.load(fp))}
        with open(os.path.join(meta_folder, f"{split}_data.json"), encoding="utf8") as fp:
            dictlist = [json.loads(line) for line in fp if line.strip()]

        txt_features, txt_to_img, skipped = [], [], 0
        for item in dictlist:
            for caption in item["captions"]:
                vec = embed(caption, word_to_index, vectors)
                if vec is None:
                    skipped += 1
                    continue
                txt_features.append(vec)
                txt_to_img.append(img_index[item["pic_id"]])  # which image row this caption describes

        np.save(os.path.join(feature_folder, f"txt_{split}.npy"), np.stack(txt_features))
        np.save(os.path.join(feature_folder, f"txt_{split}_img.npy"), np.array(txt_to_img))
        print("#{} captions: {}  (skipped {} with no known words)".format(split, len(txt_features), skipped))

    print("\nPipeline complete.")
