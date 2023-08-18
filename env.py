import gym
from gym.spaces import Box, Discrete, Dict
import numpy as np
import random
import matplotlib.pyplot as plt

class WirelessCommunicationEnv(gym.Env):
    def __init__(self, env_config):
        super(WirelessCommunicationEnv, self).__init__()
        
        self.debugging = env_config['debugging']

        self.area_side = 10  # 구역 한 변의 길이 (10km)
        self.num_base_stations = 10  # 기지국 개수
        self.base_station_radius = 3  # 기지국 셀 반경 (3km)
        self.user_speed = 300  # 시속 300km
        self.total_time = 300  # 5분 (600초)
        self.time_interval = 1  # 1초 간격

        self.base_stations = self.generate_uniform_grid_samples()
        self.user_x = None
        self.user_y = None
        self.current_time = 0

        self.action_space = Discrete(self.num_base_stations)  # 기지국 선택
        # 관측 상태: 통신 중인 BS와 BS의 SINR
        self.observation_space = Dict(
            {
                "observation": Box(
                    low=0, high=1, shape=(2, self.num_base_stations), dtype=np.float32
                )
            }
        )

    def _cal_SINR(self):
        SINRs = np.zeros(self.num_base_stations)

        for i in range(self.num_base_stations):
            # channel gain
            comm_station = self.base_stations[i]
            distance = np.sqrt((self.user_x - comm_station[0])**2 + (self.user_y - comm_station[1])**2)
            channel_quality = 1 / (distance**2)
            fading_gain = abs(self.rayleigh_fading(channel_quality))

            # 간섭 계산
            interference = 0
            for j, station in enumerate(self.base_stations):
                if i != j:
                    distance = np.sqrt((self.user_x - station[0])**2 + (self.user_y - station[1])**2)
                    channel_quality = 1 / (distance**2)
                    interference += abs(self.rayleigh_fading(channel_quality))

            SINRs[i] = fading_gain / (interference + 1e-6)  # 간섭으로 인한 분모가 0이 되는 것을 방지
        return SINRs

    def reset(self):
        self.user_x = random.uniform(0, self.area_side)
        self.user_y = random.uniform(0, self.area_side)
        self.current_time = 0
        self.info = [self.user_x, self.user_y]
        
        # 초기 선택 BS는 없음
        self.comm_indicator = -1
        self.SINRs = self._cal_SINR()
        observation = np.append(self.comm_indicator, self.SINRs)
        self.observations = observation
        #self.observations["observation"] = observation

        return self.observations, self.info  

    def step(self, action):
        self.current_time += self.time_interval

        user_x_before = self.user_x
        user_y_before = self.user_y

        self.user_x += (self.user_speed * self.time_interval / 3600) *  random.choice([-1, 0, 1])
        self.user_y += (self.user_speed * self.time_interval / 3600) *  random.choice([-1, 0, 1])
        if self.user_x > 10 or self.user_x < 0:
            self.user_x = user_x_before
        if self.user_y > 10 or self.user_y < 0:
            self.user_y = user_y_before
        self.info = [self.user_x, self.user_y]

        self.SINRs = self._cal_SINR()
        sinr = self.SINRs[action]
        
        if sinr == self.SINRs.max():
            reward = 1
        else:
            reward = 0

        if self.current_time >= self.total_time:
            done = True
        else:
            done = False

        observation = np.append(self.comm_indicator, self.SINRs)
        self.observations = observation

        if self.debugging:
            print(f"현재 time step: {self.current_time}")
            print(f"선택 기지국: {action}")
            print(f"SINRs: {observation[1:]}")
            self.render(action)

        truncated = False
        return self.observations, reward, done, truncated, self.info

    def render(self, action):
        plt.clf() 
        # 기지국 그리기
        for station in self.base_stations:
            circle = plt.Circle((station[0], station[1]), self.base_station_radius, color='gray', fill=False, linestyle='dotted')
            plt.gca().add_artist(circle)
            plt.scatter(station[0], station[1], color='blue', marker='*', s=30)
        
        # 유저와 통신 중인 기지국 연결선 그리기 
        selected_station = self.base_stations[action]
        plt.plot([self.user_x, selected_station[0]], [self.user_y, selected_station[1]], color='black', linestyle='solid')
        plt.scatter(self.user_x, self.user_y, color='red', marker='o', s=30)
        
        plt.xlim(0, self.area_side)
        plt.ylim(0, self.area_side)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.001)  # 잠시 멈추어 그래프 업데이트

    def generate_uniform_grid_samples(self):
        """
        추후 포아송으로 변경
        """
        base_stations = []
        num_rows = int(np.sqrt(self.num_base_stations))
        num_cols = int(np.ceil(self.num_base_stations / num_rows))
        x_spacing = self.area_side / (num_cols - 1)
        y_spacing = self.area_side / (num_rows - 1)

        for row in range(num_rows):
            for col in range(num_cols):
                x = col * x_spacing
                y = row * y_spacing
                base_stations.append((x, y))

        return base_stations


    def rayleigh_fading(self, channel_quality):
        h_real = np.random.normal(0, np.sqrt(channel_quality / 2))
        h_imag = np.random.normal(0, np.sqrt(channel_quality / 2))
        h = h_real + 1j * h_imag
        return h


if __name__ == '__main__':
    # Gym 환경 생성 및 테스트
    env_config = {
        'debugging': True,
    }
    env = WirelessCommunicationEnv(env_config)

    for episode in range(5):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()  # 랜덤 액션 선택
            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode+1}, Total Reward: {total_reward}")

    print("Done")