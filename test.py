import random
import matplotlib.pyplot as plt
import numpy as np

def generate_poisson_disk_samples(width, height, radius, num_attempts=30):
    cell_size = radius / (2 ** 0.5)
    grid_width = int(width / cell_size)
    grid_height = int(height / cell_size)
    
    grid = [[-1] * grid_width for _ in range(grid_height)]
    active_points = []
    samples = []
    
    initial_point = (random.uniform(0, width), random.uniform(0, height))
    active_points.append(initial_point)
    samples.append(initial_point)
    
    while active_points:
        active_index = random.randint(0, len(active_points) - 1)
        current_point = active_points[active_index]
        found = False
        
        for _ in range(num_attempts):
            angle = random.uniform(0, 2 * 3.14159)
            distance = random.uniform(radius, 2 * radius)
            new_point = (current_point[0] + distance * np.cos(angle), current_point[1] + distance * np.sin(angle))
            
            if 0 <= new_point[0] < width and 0 <= new_point[1] < height:
                grid_x = int(new_point[0] / cell_size)
                grid_y = int(new_point[1] / cell_size)
                is_valid = True
                
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        neighbor_x = grid_x + dx
                        neighbor_y = grid_y + dy
                        
                        if 0 <= neighbor_x < grid_width and 0 <= neighbor_y < grid_height:
                            neighbor = grid[neighbor_y][neighbor_x]
                            if neighbor != -1 and np.sqrt((new_point[0] - neighbor[0])**2 + (new_point[1] - neighbor[1])**2) < radius:
                                is_valid = False
                                break
                
                if is_valid:
                    samples.append(new_point)
                    active_points.append(new_point)
                    grid[grid_y][grid_x] = new_point
                    found = True
                    break
        
        if not found:
            active_points.pop(active_index)
            
    return samples

# 구역 크기 및 기지국 반경 설정
area_width = 10
area_height = 10
base_station_radius = 0.5

# Poisson Disk Sampling을 이용한 기지국 배치
base_stations = generate_poisson_disk_samples(area_width, area_height, base_station_radius)

# 결과 출력을 위한 시각화
plt.figure(figsize=(6, 6))
for station in base_stations:
    plt.scatter(station[0], station[1], color='blue', marker='o', s=30)
plt.xlim(0, area_width)
plt.ylim(0, area_height)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
