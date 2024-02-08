import sys
sys.path.append('/home/yxue/model_fusion_drl')
from copy import deepcopy
import os
import itertools
import numpy as np
import tensorflow as tf  # 不加会报错
import torch
from torch.optim import Adam
import gym
import time
import rl.sac_core as core
from spinup.utils.logx import EpochLogger  # https://spinningup.readthedocs.io/zh_CN/latest/utils/logger.html#spinup.utils.logx.EpochLogger
import logging
from common_sample_wise import CUR_DEVICE, write_results_to_file, to_tensor, to_numpy, check_exist_dir_path

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=CUR_DEVICE)
        self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=CUR_DEVICE)
        self.act_buf = torch.zeros(core.combined_shape(size, act_dim), dtype=torch.float32, device=CUR_DEVICE)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=CUR_DEVICE)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=CUR_DEVICE)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: v for k,v in batch.items()}


def sac(train_env_fn, test_env_fn=None, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, update_after=1000, update_every=50, logger_kwargs=dict(), seed=42):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.
            决定要更新网络时，需要收集些数据，这里指的是收集多少个

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
            在真实采样之前，先随机采样一段时间，由该变量定义

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    env = train_env_fn()
    test_env = test_env_fn() if test_env_fn else None
    obs_dim = env.observation_space.shape  # 特征提取器的输出维度
    act_dim = env.action_space.shape[0]  # 5

    env.seed(seed)
    env.action_space.seed(seed)
    if test_env:
        test_env.seed(seed)
        test_env.action_space.seed(seed)

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)  # 两个Critic网络和一个Actor网络
    ac_targ = deepcopy(ac)  # 是不是这里面的pi网络不用？

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    # 目标网络不是通过梯度更新的，而是跟着ac里两个Q网络走
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    # 展示网络参数量
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    
    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=to_numpy(q1),
                      Q2Vals=to_numpy(q2))

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=to_numpy(logp_pi))

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving，告诉logger要存的模型
    logger.setup_pytorch_saver(ac)
    # print(logger.pytorch_saver_elements)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(o, deterministic)
        
    def test_agent():
        print('start test')
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        # while not(d or (ep_len == max_ep_len)):  # 这里max_ep_len给的是训练集长度，所以现有d=done，即退出
        while not d:  
            # Take deterministic actions at test time 
            a = get_action(o, True)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    #### Main loop: collect experience in env and update/log each epoch
    print('interact with environment')
    for t in range(total_steps):  # 一共有5w步
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:  # start_steps步之后再开始用网络采样action
            a = get_action(o)
        else:
            a = to_tensor(env.action_space.sample())  # 1500步之前随机采一个动作

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update most recent observation!
        o = o2

        # End of trajectory handling
        # 如果在此刻的step，所有ob已经过完一遍，即训练集都用过了，就reset环境
        if d:
            logger.store(EpRet=ep_ret, EpLen=ep_len)  # 记录过这遍训练集得到的return总和，和训练集的长度
            logger.store(EpNumCor_f=env.cnt_f)
            logger.store(EpNumCor_esm=env.cnt_ensemble)
            logger.store(EpNumCor_esm_selected=env.cnt_ensemble_selected)
            logger.store(EpNumCor_all=env.cnt_all)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:  # 在2k步之后，每50步做一次更新
            print(f'Step {t}: update network')
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)  # 采样1024个记录出来更新模型
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:  # epoch结束，注意这里不是指训练集都过了一遍，而是每steps_per_epoch记录一次；如果steps_per_epoch是训练集长度，那就是每个训练集过一遍结束
            epoch = t // steps_per_epoch
            
            logger.save_state({'env': env}, itr=epoch)  # 用来保存模型的，前面有logger.setup_pytorch_saver(ac)，这里调用就会存。按这里的设置，是每个epoch存一个。保存的有环境状态.pkl和模型文件.pt

            # Test the performance of the deterministic version of the agent.
            # TTA不需要
            # t1 = time.time()
            # test_agent()
            # t2 = time.time()
            # print(f'test time: {t2-t1}')

            # log_tabular计算内部状态下每个key的平均值，标准偏差，最小值和最大值，之后清楚所有记录
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=False, average_only=True)  # 只有一个数，记录每个epoch的return
            logger.log_tabular('EpNumCor_f', with_min_and_max=False, average_only=True)
            logger.log_tabular('EpNumCor_esm', with_min_and_max=False, average_only=True)
            logger.log_tabular('EpNumCor_esm_selected', with_min_and_max=False, average_only=True)
            logger.log_tabular('EpNumCor_all', with_min_and_max=False, average_only=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=False, average_only=True)
            logger.log_tabular('EpLen', with_min_and_max=False, average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t+1)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            # 将诊断信息写入文件和标准输出
            logger.dump_tabular() 

            

