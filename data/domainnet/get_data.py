import sys
sys.path.append('/home/yxue/model_fusion_drl/data/domainnet')
from os import path
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from image_list import ImageList

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


class DomainNetLoader:
    def __init__(
        self,
        domain_name='clipart',
        dataset_path=None,
        batch_size=64,
        num_workers=4,
        use_gpu=False,
        _C=None, 
    ):
        super(DomainNetLoader, self).__init__()
        self.domain_name = domain_name
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self._C = _C
    
    def read_data(self, domain_name, split='train'):
        data_paths = []
        data_labels = []
        split_file = path.join(self.dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(self.dataset_path, data_path)
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
        print(f'==> DomainNet {domain_name} {split} {len(data_paths)} images')
        return data_paths, data_labels
    
    def read_data_multi_source(self, domain_ls, size=None, split='train'):
        img_path = []
        label_path = []

        for d_idx, d in enumerate(domain_ls):
            data_tmp, label_tmp = [], []
            split_file = path.join(self.dataset_path, "splits", "{}_{}.txt".format(d, split))
            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    data_path, label = line.split(' ')
                    data_path = path.join(self.dataset_path, data_path)
                    label = int(label)
                    data_tmp.append(data_path)
                    label_tmp.append(label)
            if size:  # 随机选出size个样本
                select_order = np.random.choice(len(data_tmp), size=size[d_idx], replace=False)
            else:  # 所有样本
                select_order = range(len(data_tmp))
            img_path.extend(data_tmp[i] for i in select_order)
            label_path.extend(label_tmp[i] for i in select_order)

        print(f'==> DomainNet {domain_ls} {split} {len(img_path)} images')
        return img_path, label_path
    

def get_domainnet126(image_root, src_domain):
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    label_file = os.path.join(image_root, f"{src_domain}_list.txt")
    test_dataset = ImageList(image_root, label_file, transform=test_transform)
    # print(len(test_dataset))
    img_path = [x[0] for x in test_dataset.item_list]
    label = [x[1] for x in test_dataset.item_list]

    return img_path, label

if __name__ == '__main__':
    get_domainnet126('/home/yxue/datasets/DomainNet-126', 'clipart')