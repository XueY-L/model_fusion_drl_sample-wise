import gym
gym.logger.set_level(40)
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import timm
from PIL import Image
import sys
sys.path.append('/home/yxue/model_fusion_drl_sample-wise')
from common_sample_wise import OPT_DIR, CUR_DEVICE, load_model_imagenetc, lerp_multi, SOURCE_DOMAIN, TARGET_DOMAIN, cal_entropy, make_custom_dataset, get_robust_bench_res50, LOSS_SCALE, load_features


class Env_train_imagenetc_sample_wise(gym.Env):
    def __init__(self):
        super(Env_train_imagenetc_sample_wise, self).__init__()
        self.num_soruce = len(SOURCE_DOMAIN.split(', '))
        self.ob_dim = 1280  # mobilenetV2提取特征是1280, vit-b是768
        self.action_space = gym.spaces.Box(np.array([0] * self.num_soruce), np.array([1] * self.num_soruce))  # 有5个源域，算weight时会softmax
        self.observation_space = gym.spaces.Box(np.array([0] * self.ob_dim), np.array([1] * self.ob_dim))
        
        # -----训练数据------
        domains = TARGET_DOMAIN.split(', ')
        self.data_path, self.label = [], []
        for d in domains:
            data_folder_path = f'/home/yxue/datasets/ImageNet-C/{d}/5'
            samples = make_custom_dataset(data_folder_path, '/home/yxue/model_fusion_drl/data/imagenetc/imagenet_test_image_ids.txt', '/home/yxue/model_fusion_drl/data/imagenetc/imagenet_class_to_id_map.json')
            paths = [s[0] for s in samples]
            labels = [s[1] for s in samples]
            self.data_path.extend(paths)
            self.label.extend(labels)
        print(len(self.data_path), len(self.label))

        # ----训练数据提取的特征----
        self.features = load_features('/home/yxue/model_fusion_drl/data/imagenetc/ImageNetC_feature_label_mobilenetv2.pt', domains, 5000)
        print(self.features.size())

        self.param_ls = load_model_imagenetc(domain_ls=SOURCE_DOMAIN.split(', '), severity=1)
        self.model = get_robust_bench_res50()
        self.loss = nn.CrossEntropyLoss()
        self.t = 0  # 时刻，表明到第t张图片了

        # 记录训练权重
        self.f = open(f'{OPT_DIR}/train_weights.txt', 'a')

    def step(self, action):
        reward = self._get_reward(action)  # t时刻动作的奖励
        self.t += 1
        if self.t % 50 == 0:
            self.f.write(f'cnt_f: {self.cnt_f} \t cnt_esm: {self.cnt_ensemble} \n')

        if self.t < len(self.data_path):
            state = self._get_next_state()  # 拿到下一时刻的状态
        else:
            state = torch.zeros(self.ob_dim, device=CUR_DEVICE)
        done = True if self.t == len(self.data_path) else False
        info = {'State': state, 'Reward': reward}
        if done:
            print('Trajecty Done')
        return state, reward, done, info

    def get_results(self, act):
        # act是weights，返回聚合后的推理logits
        weights = act.softmax(dim=-1)  # torch.size [5]
        target_param = lerp_multi(self.param_ls, weights)

        self.model.to(CUR_DEVICE)
        self.model.load_state_dict(target_param)
        self.model.eval()

        with torch.no_grad():
            rst = self.model(self.batch_data)
        
        self.f.write(f't={[self.t, self.t]}\t')
        self.f.write(f'{weights}\n')

        return rst, weights

    def _get_reward(self, act):
        rst, weights = self.get_results(act)
        weights = torch.squeeze(weights, 0)  # 在第0维的维度是1时，去掉第0维的维度
        
        # 融合模型的结果
        _, predicted_f = torch.max(rst.data, 1)
        correct_f = predicted_f.eq(self.batch_label.data).cpu().sum()
        top1_f = correct_f / rst.size(0)
        self.cnt_f += correct_f

        # ------TTA loss作为reward-----
        tent_loss = -(rst.softmax(1) * rst.log_softmax(1)).sum(1).mean(0)
        reward = -tent_loss * LOSS_SCALE

        return reward

    def _get_next_state(self):
        # 读入第t张图片，用特征提取器来提取特征，作为下一个状态
        transforms_test = transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        cnt = self.t
        self.batch_data, batch_feature, self.batch_label = [], [], []  # 这个batch的图像和标签
        while len(self.batch_data) < 1 and cnt < len(self.data_path):
            img = Image.open(self.data_path[cnt]).convert('RGB')
            img = transforms_test(img).to(CUR_DEVICE)
            self.batch_data.append(img)
            self.batch_label.append(self.label[cnt])
            batch_feature.append(self.features[cnt])
            cnt += 1
        self.batch_data = torch.stack(self.batch_data, dim=0).to(CUR_DEVICE)
        self.batch_label = torch.tensor(self.batch_label).to(CUR_DEVICE)
        batch_feature = torch.stack(batch_feature).to(CUR_DEVICE)
        batch_feature = torch.mean(batch_feature, dim=0)
        # print(self.batch_data.size(), self.batch_label.size(), batch_feature.size())
        return batch_feature

    def reset(self):
        # 重新回到第一张图片
        self.t = 0
        self.cnt_f = 0  # 融合模型分类对的数量
        self.cnt_ensemble = 0  # # 源域模型集成分类对的数量
        self.cnt_ensemble_selected = 0
        self.cnt_all = 0 
        state = self._get_next_state()
        return state

    def render(self, mode='human'):
        print(f'Step: {self.t}')



if __name__ == '__main__':
    env_train = Env_train_imagenetc_sample_wise()
    input = torch.rand(1,3,224,224).to(CUR_DEVICE)
    env_train.reset()
    print(env_train._get_reward(act=torch.tensor([0.25, 0.25, 0.25, 0.25])))