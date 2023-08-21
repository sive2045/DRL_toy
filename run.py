import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from env import WirelessCommunicationEnv

env_config = {
        'debugging': False,
    }
env = WirelessCommunicationEnv(env_config=env_config)

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = env.action_space.n
# 상태 관측 횟수를 얻습니다.
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def evaluation(path):
    # 평가 환경 설정
    env_config = {
        'debugging': True,
    }
    env = WirelessCommunicationEnv(env_config=env_config)
    num_episodes = 1
    state, info = env.reset()

    # 저장한 모델 불러오기
    model_path = path  # 모델 파일 경로
    input_size = len(state)
    output_size = env.action_space.n
    loaded_model = DQN(input_size, output_size)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    # 모델 평가
    total_rewards = 0
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        plt.ion()
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = loaded_model(state_tensor)
            action = q_values.argmax().item()

            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        plt.ioff()  # 대화형 모드 비활성화
        plt.show()  # 그래프 창 열기
        total_rewards += episode_reward
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    average_reward = total_rewards / num_episodes
    print(f"Average Reward: {average_reward}")


def run(path, test_mode):
    if test_mode:
        evaluation(path)
    else:
        if torch.cuda.is_available():
            num_episodes = 600
        else:
            num_episodes = 100
        
        episode_rewards = []
        for i_episode in range(num_episodes):
            # 환경과 상태 초기화
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = []
            total_reward = 0
            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                total_reward += reward
                episode_reward.append(total_reward)
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # 메모리에 변이 저장
                memory.push(state, action, next_state, reward)
                

                # 다음 상태로 이동
                state = next_state

                # (정책 네트워크에서) 최적화 한단계 수행
                optimize_model()

                # 목표 네트워크의 가중치를 소프트 업데이트
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    print(f"{i_episode}-에피소드, average reward: {episode_reward[-1]/env.total_time}")
                    episode_rewards.append(episode_reward[-1]/env.total_time)
                    break

        print('학습 완료')
        steps = np.arange(0,num_episodes)
        plt.plot(steps, episode_rewards)
        plt.show()
        
        print("모델 저장...")
        torch.save(target_net.state_dict(), path)



if __name__ == '__main__':
    path = '.\DQN_model.pth'
    test_mode = True # True: 모델 평가, False: 학습

    run(path=path, test_mode=test_mode)