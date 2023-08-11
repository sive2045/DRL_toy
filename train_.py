import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

from env import WirelessCommunicationEnv

# 환경 생성 함수 정의
def make_env():
    return WirelessCommunicationEnv()

# 환경 백터화
env = SubprocVectorEnv([make_env])

# DQN 네트워크 정의
class DQN(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = Net(
            [(state_shape[0],)], hidden_sizes=[128, 128], output_size=action_shape[0]
        )

    def forward(self, obs, state=None, info={}):
        return self.model(obs)

# DQN 학습 설정
state_shape = env.observation_space.shape
action_shape = env.action_space.shape
net = DQN(state_shape, action_shape)
optim = optim.Adam(net.parameters(), lr=1e-3)
policy = DQNPolicy(net, optim, nn.MSELoss(), discount_factor=0.99)

# 데이터 수집 설정
collector = Collector(policy, env)

# Replay Buffer 설정
buffer = ReplayBuffer(size=2000, stack_num=1)

# 학습 설정
trainer = offpolicy_trainer(
    policy,
    collector,
    buffer,
    max_epoch=100,
    step_per_epoch=1000,
    collect_per_step=1,
    batch_size=64,
    update_per_step=1,
    train_fn_args={'eps_min': 0.1},
)

# 학습 시작
trainer()
