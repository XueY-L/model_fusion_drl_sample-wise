import sys
sys.path.append('/home/yxue/model_fusion_drl')

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

all_data, label = load_cifar(
    data_root_dir='/home/yxue/datasets/CIFAR-100-C',
    corruptions=domains,
)
print(all_data.size(), label.size())

class DomainNetSet(Dataset):
    def __init__(self, data, data_labels, transforms):
        super(DomainNetSet, self).__init__()
        self.data = data
        self.data_labels = data_labels
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms:
            img = self.transforms(self.data[index])
        else:
            img = self.data[index]
        label = self.data_labels[index]

        return img, label

    def __len__(self):
        return len(self.data)


train_dataset = DomainNetSet(all_data, label, None)
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
    'CIFAR100C_feature_label_resnet50.pt'
)