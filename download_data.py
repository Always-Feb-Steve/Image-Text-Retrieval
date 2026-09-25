# Download Flickr8k (images + captions) and GloVe word vectors into ./data/
import os
import shutil
import urllib.request
import zipfile

# ── Paths (update these to match your environment) ──────────────────────────
data_dir = "./data"
# ─────────────────────────────────────────────────────────────────────────────

FLICKR = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/"
GLOVE = "https://nlp.stanford.edu/data/glove.6B.zip"


def download(url, dest):
    if os.path.exists(dest):
        print("Already downloaded:", dest)
        return
    print("Downloading", url)
    urllib.request.urlretrieve(url, dest + ".part")
    os.rename(dest + ".part", dest)


os.makedirs(data_dir, exist_ok=True)

# Captions + official train/dev/test split (2 MB)
if not os.path.isdir(os.path.join(data_dir, "Flickr8k_text")):
    zip_path = os.path.join(data_dir, "Flickr8k_text.zip")
    download(FLICKR + "Flickr8k_text.zip", zip_path)
    zipfile.ZipFile(zip_path).extractall(os.path.join(data_dir, "Flickr8k_text"))
    os.remove(zip_path)

# 8,091 images (1.1 GB); the archive's folder name really is "Flicker8k_Dataset"
if not os.path.isdir(os.path.join(data_dir, "Flicker8k_Dataset")):
    zip_path = os.path.join(data_dir, "Flickr8k_Dataset.zip")
    download(FLICKR + "Flickr8k_Dataset.zip", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir, [m for m in zf.namelist() if m.startswith("Flicker8k_Dataset/")])
    os.remove(zip_path)

# GloVe 6B (860 MB zip); only the 200-dim vectors are kept
glove_txt = os.path.join(data_dir, "glove.6B.200d.txt")
if not os.path.exists(glove_txt):
    zip_path = os.path.join(data_dir, "glove.6B.zip")
    download(GLOVE, zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open("glove.6B.200d.txt") as src, open(glove_txt, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(zip_path)

print("Data ready in", data_dir)
