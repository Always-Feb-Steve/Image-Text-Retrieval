# Extract image features using ResNet50
import json
import os
import time

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── Paths (update these to match your environment) ──────────────────────────
meta_folder     = "./meta_data/"
pic_folder      = "./data/Flicker8k_Dataset/"
feature_folder  = "./features/"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(feature_folder, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ResNet50 pre-trained on ImageNet. Replacing the final classification layer with an
# identity makes the network return the 2048-dim output of the "avgpool" layer directly.
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet50.fc = nn.Identity()
resnet50.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # ResNet50 was trained on ImageNet-normalized inputs
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PicDataset(Dataset):
    def __init__(self, pic_ids):
        self.pic_ids = pic_ids

    def __len__(self):
        return len(self.pic_ids)

    def __getitem__(self, index):
        img = Image.open(os.path.join(pic_folder, self.pic_ids[index] + ".jpg")).convert("RGB")
        return transform(img)


def process(split, device):
    """Extract ResNet50 features for every image in one split and save them as one matrix."""
    with open(os.path.join(meta_folder, f"{split}_data.json"), encoding="utf8") as fp:
        pic_ids = [json.loads(line)["pic_id"] for line in fp if line.strip()]
    print("#{} images: {}".format(split, len(pic_ids)))

    loader = DataLoader(PicDataset(pic_ids), batch_size=64, num_workers=4)
    features = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            features.append(resnet50(batch.to(device)).cpu().numpy())
    features = np.concatenate(features).astype(np.float32)  # (N, 2048)
    print("  extracted {} in {:.1f}s".format(features.shape, time.perf_counter() - t0))

    # Row i of img_<split>.npy belongs to pic_ids[i]
    np.save(os.path.join(feature_folder, f"img_{split}.npy"), features)
    with open(os.path.join(feature_folder, f"img_{split}_ids.json"), "w", encoding="utf8") as fp:
        json.dump(pic_ids, fp)


if __name__ == "__main__":
    device = get_device()
    print("Using device:", device)
    resnet50.to(device)
    for split in ["train", "val", "test"]:
        process(split, device)
