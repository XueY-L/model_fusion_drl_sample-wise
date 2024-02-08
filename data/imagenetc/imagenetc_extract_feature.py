import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm
from PIL import Image
from common import BATCH_SIZE, LOSS_SCALE, OPT_DIR, CUR_DEVICE, load_model_cifar100c, load_cifar, lerp_multi, SOURCE_DOMAIN, TARGET_DOMAIN, cal_entropy, get_Hendrycks_AugMixResNeXtNet, make_custom_dataset

domains = 'gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression'.split(', ')

data_path, label = [], []
for d in domains:
    data_folder_path = f'/home/yxue/datasets/ImageNet-C/{d}/5'
    samples = make_custom_dataset(data_folder_path, 'data/imagenet_test_image_ids.txt', 'data/imagenet_class_to_id_map.json')
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    data_path.extend(paths)
    label.extend(labels)
print(len(data_path), len(label))


class DomainNetSet(Dataset):
    def __init__(self, data_paths, data_labels, transforms):
        super(DomainNetSet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)

transforms_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
train_dataset = DomainNetSet(data_path, label, transforms_test)
train_dloader = DataLoader(train_dataset, batch_size=250, num_workers=16, pin_memory=True, shuffle=False)

# feature_model = models.mobilenet_v2(pretrained=True).to(CUR_DEVICE)
# feature_model.classifier = nn.Sequential()
# dim = 1280
feature_model = models.resnet50(pretrained=True).to(CUR_DEVICE)
feature_model.fc = nn.Sequential()
dim = 2048

feature_model.eval()
features = []
with torch.no_grad():
    for data, _ in train_dloader:
        data = data.to(CUR_DEVICE)
        out = feature_model(data)  # torch.Size([bs, 1280])
        print(data.size(), out.size())
        features.append(out)
features = torch.stack(features)
features = features.view(-1, dim)
print(features.size())

torch.save(
    {
        'data': features,
        'label': label, 
    }, 
    'ImageNetC_feature_label_resnet50.pt'
)