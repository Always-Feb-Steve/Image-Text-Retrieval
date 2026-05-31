# Extract image features using ResNet50
import torch
import json
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
import PIL.Image as Image
from multiprocessing import Pool
import time
import os


begintime = time.perf_counter()

resnet50 = models.resnet50(pretrained=True)  # Initialize ResNet50 for extracting embeddings (50 layers)
resnet50.eval()

extract_list = ["avgpool"]


class FeatureExtractor(nn.Module):
    """Extracts intermediate layer features from a given submodule."""

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule          # ResNet50 model
        self.extracted_layers = extracted_layers  # e.g. ['avgpool'] — returns embedding after this layer

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# ── Paths (update these to match your environment) ──────────────────────────
train_json_path = "./meta_data/train_data.json"
test_json_path  = "./meta_data/test_data.json"
pic_folder               = "./pics/"
save_pic_feature_folder  = "./train_pic_npy/"
test_save_pic_feature_folder = "./test_pic_npy/"
save_id_npy_folder       = "./inverted_index/"
# ─────────────────────────────────────────────────────────────────────────────

for folder in [save_pic_feature_folder, test_save_pic_feature_folder, save_id_npy_folder]:
    os.makedirs(folder, exist_ok=True)


def extract_picid_feature(one_item):
    """Extract and save ResNet50 feature for a single training image."""
    pic_id   = str(one_item["pic_id"])
    img_path = os.path.join(pic_folder, pic_id + ".jpg")
    pure_id  = pic_id.split('.')[0]
    save_path = os.path.join(save_pic_feature_folder, pure_id + ".npy")

    if os.path.exists(save_path):
        print("Already processed:", pic_id)
        return pic_id

    try:
        img = transform(Image.open(img_path))
    except Exception:
        print("Error reading image:", pic_id)
        return None

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    extractor = FeatureExtractor(resnet50, extract_list)

    try:
        savebuffer = extractor(x)[0].detach().numpy()
    except Exception:
        print("Error extracting features for image:", pic_id)
        return None

    np.save(save_path, savebuffer)
    return pic_id


def extract_picid_feature_test(one_item):
    """Extract and save ResNet50 feature for a single test image."""
    pic_id   = str(one_item["pic_id"])
    img_path = os.path.join(pic_folder, pic_id + ".jpg")
    pure_id  = pic_id.split('.')[0]
    save_path = os.path.join(test_save_pic_feature_folder, pure_id + ".npy")

    if os.path.exists(save_path):
        return pic_id

    try:
        img = transform(Image.open(img_path))
    except Exception:
        print("Error reading image:", pic_id)
        return None

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    extractor = FeatureExtractor(resnet50, extract_list)

    try:
        savebuffer = extractor(x)[0].detach().numpy()
    except Exception:
        print("Error extracting features for image:", pic_id)
        return None

    np.save(save_path, savebuffer)
    return pic_id


def process_train():
    """Process all training images and save their feature embeddings."""
    with open(train_json_path, "rb") as fp:
        buffer = fp.read()

    dictlist = []
    for item in str(buffer, encoding="UTF-8").split("\n"):
        try:
            dictlist.append(json.loads(item))
        except Exception:
            continue
    print("#Train samples:", len(dictlist))

    pool = Pool(processes=10)
    res  = pool.map(extract_picid_feature, dictlist)
    pool.close()
    pool.join()

    new_dict = [r for r in res if r is not None]

    with open(os.path.join(save_id_npy_folder, "train_img_npy_id.json"), "w", encoding="UTF-8") as fp:
        json.dump(new_dict, fp, ensure_ascii=False)


def process_test():
    """Process all test images and save their feature embeddings."""
    with open(test_json_path, "rb") as fp:
        buffer = fp.read()

    dictlist = []
    for item in str(buffer, encoding="UTF-8").split("\n"):
        try:
            dictlist.append(json.loads(item))
        except Exception:
            continue
    print("#Test samples:", len(dictlist))

    pool = Pool(processes=10)
    res  = pool.map(extract_picid_feature_test, dictlist)
    pool.close()
    pool.join()

    new_dict = [r for r in res if r is not None]

    with open(os.path.join(save_id_npy_folder, "test_img_npy_id.json"), "w", encoding="UTF-8") as fp:
        json.dump(new_dict, fp, ensure_ascii=False)


if __name__ == "__main__":
    process_train()
    process_test()
