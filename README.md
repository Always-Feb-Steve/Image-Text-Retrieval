# Image-Text Retrieval Search Engine

A cross-modal image retrieval system: given a text keyword, retrieve and display matching images. The pipeline builds an inverted index from metadata, extracts image and text embeddings, trains a cross-modal ranking model, and serves results via a Flask web interface.

## Project Structure

```
image-text-retrieval/
├── meta_data/
│   ├── input.json          # Raw metadata (tags + image IDs)
│   ├── output.json         # Inverted index (tag -> [pic_ids])
│   └── word2vec.utf8       # Pre-trained word vectors (200-dim)
├── pics/                   # Raw image files (not tracked by git)
├── inoutput_json.py        # Step 1: Build inverted index
├── extract_img.py          # Step 2a: Extract ResNet50 image features
├── extract_txt.py          # Step 2b: Extract Word2Vec text features
├── intersection.py         # Step 3: Align image and text IDs
├── rank.py                 # Step 4: Train cross-modal ranking model
├── Flask_Web.py            # Step 5: Serve search engine via Flask
├── requirements.txt
└── .gitignore
```

## Pipeline Overview

```
input.json
     │
inoutput_json.py       → output.json (inverted index: tag -> [pic_ids])
     │
     ├── extract_img.py  → train_pic_npy/  test_pic_npy/   (ResNet50 embeddings)
     └── extract_txt.py  → train_text_npy/ test_text_npy/  (Word2Vec embeddings)
                │
           intersection.py  → train_npy_id.json / test_npy_id.json
                │
             rank.py        → models/net.pkl  (trained cross-modal ranker)
                │
           Flask_Web.py     → http://localhost:5000/SearchEngine/?tag=<keyword>
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the inverted index

Reads `meta_data/input.json` and writes `meta_data/output.json`:

```bash
python inoutput_json.py
```

### 3. Extract features

**Image features** (ResNet50 `avgpool` layer, 2048-dim):
```bash
python extract_img.py
```

**Text features** (Word2Vec lookup, 200-dim):
```bash
python extract_txt.py
```

### 4. Align IDs

Find the intersection of images and texts that both have valid features:
```bash
python intersection.py
```

### 5. Train the ranking model

Trains a two-branch fully-connected network with hinge loss to align image and text embeddings in a shared 128-dim space:
```bash
python rank.py
```

Best checkpoint saved to `models/net.pkl.best.pkl`.

### 6. Run the search engine

```bash
python Flask_Web.py
```

Then open your browser at:
```
http://localhost:5000/SearchEngine/?tag=<your_keyword>
```

## Model Architecture

```
Image (2048-dim)  ──→  Linear(2048, 128)  ──→  embedding
Text  (200-dim)   ──→  Linear(200,  128)  ──→  embedding
                                │
                     Cosine Similarity
                                │
                        Hinge Loss (margin=0.2)
```

The model is trained with triplet-style ranking: for each sample, a positive image (matching the text) and a negative image (offset by 3) are passed through the network. The loss pushes the positive pair's cosine similarity above the negative pair's by at least the margin.

## Data Format

`input.json` — one JSON object per line:
```json
{"pic_id": 0, "tags_term": "some_tag", "pic_url": "https://..."}
```

`output.json` — inverted index:
```json
{"some_tag": [0, 1, 2, ...], "other_tag": [5, 6, ...]}
```

## Notes

- Update the path constants at the top of each script to match your local directory layout.
- `.npy` feature files, model weights, and raw images are excluded from git via `.gitignore`.
- The `pics/` folder should contain images named `{pic_id}.jpg`.
