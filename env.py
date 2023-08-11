import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

class WirelessCommunicationEnv(gym.Env):
    def __init__(self):
        super(WirelessCommunicationEnv, self).__init__()

        self.area_side = 10  # 구역 한 변의 길이 (10km)
        self.num_base_stations = 10  # 기지국 개수
        self.base_station_radius = 3  # 기지국 셀 반경 (3km)
        self.user_speed = 80  # 시속 80km
        self.total_time = 600  # 10분 (600초)
        self.time_interval = 1  # 1초 간격

        self.base_stations = self.generate_uniform_grid_samples()
        self.user_x = None
        self.user_y = None
        self.current_time = 0

        self.action_space = spaces.Discrete(self.num_base_stations)  # 기지국 선택
        # 관측 상태: 통신 중인 BS와 BS의 SINR
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0, high=1, shape=(2, self.num_base_stations), dtype=np.int8
                )
            }
        )

    def reset(self):
        self.user_x = random.uniform(0, self.area_side)
        self.user_y = random.uniform(0, self.area_side)
        self.current_time = 0
        info = [self.user_x, self.user_y]
        return self.num_base_stations, info  # 초기 상태는 기지국을 선택하지 않음을 나타냄

    def step(self, action):
        self.current_time += self.time_interval

        user_x_before = self.user_x
        user_y_before = self.user_y

        self.user_x += (self.user_speed * self.time_interval / 3600) * np.cos(np.radians(random.uniform(0, 360)))
        self.user_y += (self.user_speed * self.time_interval / 3600) * np.sin(np.radians(random.uniform(0, 360)))

        selected_station = self.base_stations[action]
        distance = np.sqrt((self.user_x - selected_station[0])**2 + (self.user_y - selected_station[1])**2)
        channel_quality = 1 / (distance**2)
        fading_gain = abs(self.rayleigh_fading(channel_quality))

        interference = 0
        for i, station in enumerate(self.base_stations):
            if i != action:
                distance = np.sqrt((self.user_x - station[0])**2 + (self.user_y - station[1])**2)
                channel_quality = 1 / (distance**2)
                interference += abs(self.rayleigh_fading(channel_quality))

        sinr = fading_gain / (interference + 1e-6)  # 간섭으로 인한 분모가 0이 되는 것을 방지
        reward = sinr

        if self.current_time >= self.total_time:
            done = True
        else:
            done = False

        return action, reward, done, {}

    def render(self):
        plt.figure(figsize=(6, 6))
        
        # 기지국 그리기
        for station in self.base_stations:
            circle = plt.Circle((station[0], station[1]), self.base_station_radius, color='gray', fill=False, linestyle='dotted')
            plt.gca().add_artist(circle)
            plt.scatter(station[0], station[1], color='blue', marker='*', s=30)
        
        # 유저와 통신 중인 기지국 연결선 그리기
        action, _, _, _ = self.step(self.action_space.sample())
        selected_station = self.base_stations[action]
        plt.plot([self.user_x, selected_station[0]], [self.user_y, selected_station[1]], color='black', linestyle='solid')
        plt.scatter(self.user_x, self.user_y, color='red', marker='o', s=30)
        
        plt.xlim(0, self.area_side)
        plt.ylim(0, self.area_side)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

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
    env = WirelessCommunicationEnv()

    for episode in range(5):
        state = env.reset()
        total_reward = 0
        done = False

        #env.render()
        while not done:
            action = env.action_space.sample()  # 랜덤 액션 선택
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
        #env.render()

        print(f"Episode {episode+1}, Total Reward: {total_reward}")

    print("Done")