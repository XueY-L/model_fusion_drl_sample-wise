import random
import torch
import os
import numpy as np
from rl.sac import sac
from rl.ppo import ppo
from common_sample_wise import OPT_DIR, check_exist_dir_path, SEED, TARGET_DOMAIN, REPLAY_BS, LR, EXP_NAME

from env.env_train_imagenetc_sample_wise import Env_train_imagenetc_sample_wise

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

train_env = Env_train_imagenetc_sample_wise
steps_per_epoch = len(TARGET_DOMAIN.split(', '))*5000

opt_directory = os.path.join(OPT_DIR, EXP_NAME)
try:
    check_exist_dir_path(opt_directory)
except:
    os.makedirs(opt_directory)

logger_kwargs = {'output_dir': opt_directory, 'exp_name':EXP_NAME}

if EXP_NAME == 'sac':
    from common_sample_wise import START_STEPS, UPDATE_AFTER, UPDATE_EVERY, GAMMA, REPLAY_SIZE
    sac(
        train_env_fn=train_env, test_env_fn=None,  
        ac_kwargs={}, 
        steps_per_epoch=steps_per_epoch,  # DomainNet: LEN_SET[TARGET_DOMAIN][2]; Cifar: len(TARGET_DOMAIN.split(', '))*10000
        epochs=1, 
        replay_size=REPLAY_SIZE, 
        gamma=GAMMA, 
        polyak=0.995, 
        lr=LR, 
        alpha=0.2, 
        batch_size=REPLAY_BS, 
        start_steps=START_STEPS, 
        update_after=UPDATE_AFTER, 
        update_every=UPDATE_EVERY, 
        logger_kwargs=logger_kwargs,
        seed=SEED
    )
