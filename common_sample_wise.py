import copy
import json
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tmodels
from collections import OrderedDict
from model.resnext import CifarResNeXt, ResNeXtBottleneck
from model.wide_resnet import WideResNet


EXP_NAME = 'sac'  # sac, ppo
DATASET = 'ImageNetC'  # CIFAR100C, ImageNetC, DomainNet126

CUR_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

if DATASET in ['ImageNetC', 'CIFAR100C']:
    SEVERITY = 1
    SOURCE_DOMAIN = 'gaussian_noise, glass_blur, snow, jpeg_compression'
    TARGET_DOMAIN = 'gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression'

    if DATASET == 'ImageNetC':
        LOSS_SCALE = 10
        REPLAY_SIZE = 500  # 1000000
        REPLAY_BS = 50  # 1024
    elif DATASET == 'CIFAR100C':
        LOSS_SCALE = 1
        REPLAY_SIZE = 1024  # 1000000
        REPLAY_BS = 1024  # 1024

SEED = 3407

if EXP_NAME == 'sac':
    START_STEPS = 0
    UPDATE_AFTER = 0
    UPDATE_EVERY = 10
    LR = 1e-4  # 5e-4
    GAMMA = 0.9  # 0.9

    NAME = f'Sample-wise_{DATASET}_sac_{LOSS_SCALE}loss_({SOURCE_DOMAIN}){SEVERITY}_ReP{REPLAY_SIZE, REPLAY_BS}'

WORK_DIR = 'results_all/'
OPT_DIR = os.path.join(WORK_DIR, NAME)

LEN_SET_DomainNet = {
    'clipart':[26681,6844,14604],
    'infograph':[28678,7345,15582],
    'painting':[40196,10220,21850],
    'quickdraw':[96600,24150,51750],
    'real':[96589, 24317, 52041],
    'sketch':[38433,9779,20916],
}

LEN_SET_DomainNet126 = {
    'clipart':18523,
    'painting':30042,
    'real':69622,
    'sketch':24147,
}

def json_load(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res

def json_dump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)
    
def to_numpy(var, gpu_used=True):
    return var.cpu().data.numpy() if gpu_used else var.data.numpy()

def to_tensor(ndarray, gpu_0=CUR_DEVICE, gpu_used=True):
    if gpu_used:
        return torch.from_numpy(ndarray).to(device=gpu_0).type(torch.float32)
    else:
        return torch.from_numpy(ndarray).type(torch.float32)

def check_exist_dir_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def check_exist_file_path(path):
    if os.path.exists(path):
        os.remove(path)

def safe_divide(a, b):
    if b == 0:
        return 0
    else:
        return a / b

def write_results_to_file(results, file_path):
    f = open(file_path, 'w')
    for r in results:
        f.write(f"{r.data}")
    f.close()

def load_model_domainnet(domain_ls:list):
    model_path = {  # pretrained resnet50-lr0.001
        'clipart':'/home/yxue/model_fusion_dnn/ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_clipart__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'infograph':'/home/yxue/model_fusion_dnn/ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_infograph__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'painting':'/home/yxue/model_fusion_dnn/ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_painting__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'quickdraw':'/home/yxue/model_fusion_dnn/ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_quickdraw__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'real':'/home/yxue/model_fusion_dnn/ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_real__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
        'sketch':'/home/yxue/model_fusion_dnn/ckpt_res50/checkpoint/domainnet_domainbed_hparam/hparam_lr/ckpt_sketch__sgd_lr-s0.001_lr-w0.0005_bs32_seed42_source-[]_DomainNet_resnet50-1.0x_singletraining-domainbedhparam_lrd-[-2, -1]_wd-0.0005.pth',
    }

    param_ls = []
    for d in domain_ls:
        ttt = torch.load(model_path[d], map_location=CUR_DEVICE)
        print(f'==> Loading {d} Model', '\tepoch:', ttt['epoch'], '\tAcc:', ttt['acc'].item())
        param_ls.append(ttt['net'])
    return param_ls

def load_model_domainnet126(domain_ls:list):
    model_path = {
        'clipart':'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_clipart__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth',
        'painting':'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_painting__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth',
        'real':'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_real__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth',
        'sketch':'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_sketch__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth',
    }

    param_ls = []
    for d in domain_ls:
        ttt = torch.load(model_path[d], map_location=CUR_DEVICE)
        print(f'==> Loading {d} Model', '\tepoch:', ttt['epoch'], '\tAcc:', ttt['acc'].item())
        param_ls.append(ttt['net'])
    return param_ls

def load_model_imagenetc(domain_ls, severity):
    model_path_severity5 = {
        'gaussian_noise': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'gaussian_noise\']_[5].pt',
        'shot_noise': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'shot_noise\']_[5].pt',
        'impulse_noise': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'impulse_noise\']_[5].pt',
        'defocus_blur': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'defocus_blur\']_[5].pt',
        'glass_blur': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'glass_blur\']_[5].pt', 
        'motion_blur': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'motion_blur\']_[5].pt', 
        'zoom_blur': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'zoom_blur\']_[5].pt',
        'frost': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'frost\']_[5].pt',
        'snow': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'snow\']_[5].pt',
        'fog': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'fog\']_[5].pt',
        'brightness': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'brightness\']_[5].pt',
        'contrast': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'contrast\']_[5].pt',
        'elastic_transform': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'elastic_transform\']_[5].pt',
        'pixelate': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'pixelate\']_[5].pt',
        'jpeg_compression': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'jpeg_compression\']_[5].pt',
    }
    model_path_severity3 = {
        'gaussian_noise': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'gaussian_noise\']_[3].pt',
        'glass_blur': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'glass_blur\']_[3].pt', 
        'snow': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'snow\']_[3].pt',
        'jpeg_compression': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'jpeg_compression\']_[3].pt',
    }
    model_path_severity1 = {
        'gaussian_noise': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'gaussian_noise\']_[1].pt',
        'glass_blur': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'glass_blur\']_[1].pt', 
        'snow': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'snow\']_[1].pt',
        'frost': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'frost\']_[1].pt',
        'fog': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'fog\']_[1].pt',
        'brightness': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'brightness\']_[1].pt',
        'jpeg_compression': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'jpeg_compression\']_[1].pt',
    }
    param_ls = []
    for d in domain_ls:
        if d == 'clean':
            temp = nn.Sequential(
                OrderedDict([
                    ('normalize', ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
                    ('model', tmodels.resnet50(pretrained=True)),
                ])
            )
            param_ls.append(temp.state_dict())
            continue
        if severity == 5:
            ttt = torch.load(model_path_severity5[d], map_location=CUR_DEVICE)
        elif severity == 3:
            ttt = torch.load(model_path_severity3[d], map_location=CUR_DEVICE)
        elif severity == 1:
            ttt = torch.load(model_path_severity1[d], map_location=CUR_DEVICE)
        print(f'==> Loading {d} Model', '\tepoch:', ttt['epoch'], '\tAcc:', ttt['acc'].item())
        param_ls.append(ttt['model'])
    return param_ls

def load_model_imagenetc_5epochs(domain_ls, severity):
    '''
    只微调了5个epochs的拉跨源模型
    '''
    model_path_severity5 = {
        
    }
    model_path_severity3 = {
        
    }
    model_path_severity1 = {
        'gaussian_noise': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'gaussian_noise\']_[1]_5epochs.pt',
        'glass_blur': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'glass_blur\']_[1]_5epochs.pt', 
        'snow': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'snow\']_[1]_5epochs.pt',
        'jpeg_compression': '/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'jpeg_compression\']_[1]_5epochs.pt',
    }
    param_ls = []
    for d in domain_ls:
        if d == 'clean':
            temp = nn.Sequential(
                OrderedDict([
                    ('normalize', ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
                    ('model', tmodels.resnet50(pretrained=True)),
                ])
            )
            param_ls.append(temp.state_dict())
            continue
        if severity == 5:
            ttt = torch.load(model_path_severity5[d], map_location=CUR_DEVICE)
        elif severity == 3:
            ttt = torch.load(model_path_severity3[d], map_location=CUR_DEVICE)
        elif severity == 1:
            ttt = torch.load(model_path_severity1[d], map_location=CUR_DEVICE)
        print(f'==> Loading {d} Model', '\tepoch:', ttt['epoch'], '\tAcc:', ttt['acc'].item())
        param_ls.append(ttt['model'])
    return param_ls

def load_model_cifar100c(domain_ls, severity):
    model_path_severity5 = {
        'gaussian_noise': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'gaussian_noise\']_[5]_noaug.pt',
        'shot_noise': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'shot_noise\']_[5]_noaug.pt',
        'impulse_noise': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'impulse_noise\']_[5]_noaug.pt',
        'defocus_blur': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'defocus_blur\']_[5]_noaug.pt',
        'glass_blur': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'glass_blur\']_[5]_noaug.pt', 
        'motion_blur': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'motion_blur\']_[5]_noaug.pt', 
        'zoom_blur': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'zoom_blur\']_[5]_noaug.pt',
        'frost': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'frost\']_[5]_noaug.pt',
        'snow': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'snow\']_[5]_noaug.pt',
        'fog': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'fog\']_[5]_noaug.pt',
        'brightness': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'brightness\']_[5]_noaug.pt',
        'contrast': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'contrast\']_[5]_noaug.pt',
        'elastic_transform': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'elastic_transform\']_[5]_noaug.pt',
        'pixelate': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'pixelate\']_[5]_noaug.pt',
        'jpeg_compression': '/home/yxue/model_fusion_tta/cifar/checkpoint/cifar100c/ckpt_[\'jpeg_compression\']_[5]_noaug.pt',
    }
    
    model_path_severity1 = {
        'gaussian_noise': '/home/yxue/model_fusion_tta/cifar/checkpoint/ckpt_cifar100_[\'gaussian_noise\']_[1].pt',
        'glass_blur': '/home/yxue/model_fusion_tta/cifar/checkpoint/ckpt_cifar100_[\'glass_blur\']_[1].pt',
        'snow': '/home/yxue/model_fusion_tta/cifar/checkpoint/ckpt_cifar100_[\'snow\']_[1].pt',
        'jpeg_compression': '/home/yxue/model_fusion_tta/cifar/checkpoint/ckpt_cifar100_[\'jpeg_compression\']_[1].pt',
    }

    param_ls = []
    for d in domain_ls:
        if severity == 5:
            ttt = torch.load(model_path_severity5[d], map_location=CUR_DEVICE)
        elif severity == 1:
            ttt = torch.load(model_path_severity1[d], map_location=CUR_DEVICE)
        print(f'==> Loading {d} Model', '\tepoch:', ttt['epoch'], '\tAcc:', ttt['acc'].item())
        param_ls.append(ttt['model'])
    return param_ls



def lerp_multi(param_ls: list, weights=None):
    if weights == None:
        weights = [1/len(param_ls) for _ in param_ls]
    weights = torch.squeeze(weights, 0)  # 在第0维的维度是1时，去掉第0维的维度
    # print(f"weights: {weights}")

    target_net = dict()
    for k in param_ls[0]:
        if 'running' in k:  # running_mean, running_var  requires_grad=True
            fs = torch.zeros(param_ls[0][k].size(), requires_grad=False).to(CUR_DEVICE)
            for idx, net in enumerate(param_ls):
                fs = fs + net[k].data * weights[idx].data
        else:
            fs = torch.zeros(param_ls[0][k].size(), requires_grad=True).to(CUR_DEVICE)
            for idx, net in enumerate(param_ls):
                fs = fs + net[k] * weights[idx]
        target_net[k] = fs
    return target_net

def cal_entropy(logits):
    if len(logits.size()) == 1:  # 如果logits是一维的，加一维
        logits = torch.unsqueeze(logits, dim=0)
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean(0)

def make_custom_dataset(root, path_imgs, cls_dict):
    with open(path_imgs, 'r') as f:
        fnames = f.readlines()
    with open(cls_dict, 'r') as f:
        class_to_idx = json.load(f)
    images = [(os.path.join(root, c.split('\n')[0]), class_to_idx[c.split('/')[0]]) for c in fnames]

    return images


class ImageNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, inp):
        if isinstance(inp, tuple):
            return ((inp[0] - self.mean) / self.std, inp[1])
        else:
            return (inp - self.mean) / self.std

def get_robust_bench_res50():
    layers = OrderedDict([
        ('normalize', ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
        ('model', tmodels.resnet50(pretrained=True))
    ])
    return nn.Sequential(layers)


##### cifar100c
def load_cifar(data_root_dir, corruptions, severity=5, n_total_cifar=10000):
    labels_path = os.path.join(data_root_dir, 'labels.npy')
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    for corruption in corruptions:
        print(f'Loading Cifar100-C {corruption}')
        corruption_file_path = os.path.join(data_root_dir, corruption+'.npy')
        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar]
        x_test_list.append(images)
        y_test_list.append(labels[:n_total_cifar])
    
    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_test = x_test.astype(np.float32) / 255
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)

    return x_test, y_test

def load_features(feature_path, corruptions, n_total):
    all_doamins = 'gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression'.split(', ')
    features = torch.load(feature_path, map_location='cpu')['data']
    features_selected = []
    for corruption in corruptions:
        idx = all_doamins.index(corruption)
        # print(idx*n_total, (idx+1)*n_total)
        temp = features[idx*n_total:(idx+1)*n_total]
        features_selected.append(temp)
    features_selected = torch.stack(features_selected)
    features_selected = features_selected.view(-1, features_selected.size(-1))
    return features_selected


def get_Hendrycks_AugMixResNeXtNet():
    def rm_substr_from_state_dict(state_dict, substr):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if substr in key:  # to delete prefix 'module.' if it exists
                new_key = key[len(substr):]
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        return new_state_dict
    
    class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):
        def __init__(self, depth=29, cardinality=4, base_width=32):
            super().__init__(ResNeXtBottleneck,
                            depth=depth,
                            num_classes=100,
                            cardinality=cardinality,
                            base_width=base_width)
            self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
            self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mu) / self.sigma
            return super().forward(x)
    
    checkpoint = torch.load('/home/yxue/model_fusion_drl/model/Hendrycks2020AugMix_ResNeXt.pt', map_location='cpu')
    state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
    
    model = Hendrycks2020AugMixResNeXtNet()
    model.load_state_dict(state_dict, strict=False)  # Missing key(s) in state_dict: "mu", "sigma". 
    
    return model

def load_standard_WRN():
    model = WideResNet(depth=28, widen_factor=10)
    checkpoint = torch.load('/home/yxue/model_fusion_drl/model/Standard.pt')['state_dict']
    model.load_state_dict(checkpoint)
    return model


if __name__ == '__main__':
    # TTA_update(torch.tensor([[1.,3.,5.], [2.,4.,6.]]), torch.randn(100, 10, 3, 224, 224), None, None)
    model = tmodels.resnet50()
    setup_model(model)