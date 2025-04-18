import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
from typing import Dict, List, Tuple, Set
import random
from dataclasses import dataclass
from copy import deepcopy
import json
import csv
from datetime import datetime, timedelta

@dataclass
class Location:
    name: str
    coordinates: Tuple[float, float]
    demand: int  # Negative for shortage, positive for surplus
    ideal_count: int = 0
    current_count: int = 0
    time_window: Tuple[datetime, datetime] = None  # 时间窗口约束
    priority: int = 1  # 优先级，数字越大优先级越高

class GeneticAlgorithm:
    def __init__(self, scheduler, population_size=50, generations=100):
        self.scheduler = scheduler
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        
    def create_individual(self):
        """创建一个个体（一组路线）"""
        all_locations = list(self.scheduler.locations.keys())
        routes = []
        for _ in range(self.scheduler.vehicle_count):
            route_length = random.randint(2, min(4, len(all_locations)))
            route = random.sample(all_locations, route_length)
            routes.append(route)
        return routes
    
    def fitness(self, individual):
        """计算适应度"""
        total_supplied = 0
        total_time = 0
        for route in individual:
            time, supplied = self.scheduler.evaluate_route(route)
            total_supplied += supplied
            total_time += time
        return total_supplied - 0.1 * total_time  # 平衡供应量和时间
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        child = []
        for route1, route2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append(route1)
            else:
                child.append(route2)
        return child
    
    def mutate(self, individual):
        """变异操作"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                route = individual[i]
                if len(route) > 2:
                    # 随机交换两个位置
                    idx1, idx2 = random.sample(range(len(route)), 2)
                    route[idx1], route[idx2] = route[idx2], route[idx1]
        return individual
    
    def optimize(self):
        """运行遗传算法"""
        population = [self.create_individual() for _ in range(self.population_size)]
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # 评估适应度
            fitness_scores = [(self.fitness(ind), ind) for ind in population]
            fitness_scores.sort(reverse=True)
            
            # 更新最佳个体
            if fitness_scores[0][0] > best_fitness:
                best_fitness = fitness_scores[0][0]
                best_individual = fitness_scores[0][1]
            
            # 选择父代
            parents = fitness_scores[:self.population_size//2]
            
            # 创建新一代
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1[1], parent2[1])
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
            
        return best_individual

class BicycleScheduler:
    def __init__(self, 
                 locations: Dict[str, Location],
                 vehicle_count: int = 3,
                 vehicle_capacity: int = 20,
                 vehicle_speed: float = 25.0,  # km/h
                 max_time: float = 1.0):  # hours
        self.locations = locations
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_speed = vehicle_speed
        self.max_time = max_time
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """计算所有位置之间的距离矩阵"""
        distances = {}
        for loc1 in self.locations:
            for loc2 in self.locations:
                if loc1 != loc2:
                    dist = np.linalg.norm(
                        np.array(self.locations[loc1].coordinates) - 
                        np.array(self.locations[loc2].coordinates)
                    )
                    distances[(loc1, loc2)] = dist
        return distances
    
    def calculate_route_time(self, route: List[str]) -> float:
        """计算给定路线的总时间"""
        total_time = 0
        for i in range(len(route) - 1):
            total_time += self.distance_matrix[(route[i], route[i + 1])] / self.vehicle_speed
        return total_time
    
    def evaluate_route(self, route: List[str]) -> Tuple[float, int]:
        """评估路线并返回(时间, 供应量)"""
        time = self.calculate_route_time(route)
        load = 0
        total_supplied = 0
        
        for i in range(len(route) - 1):
            current_loc = self.locations[route[i + 1]]
            
            # 检查时间窗口约束
            if current_loc.time_window:
                current_time = datetime.now() + timedelta(hours=time)
                if not (current_loc.time_window[0] <= current_time <= current_loc.time_window[1]):
                    return float('inf'), 0
            
            if current_loc.demand < 0:  # 短缺点
                supplied = min(-current_loc.demand, self.vehicle_capacity - load)
                total_supplied += supplied * current_loc.priority  # 考虑优先级
                load += supplied
            else:  # 富余点
                load = max(0, load - min(current_loc.demand, load))
                
        return time, total_supplied
    
    def find_optimal_routes(self) -> List[List[str]]:
        """使用遗传算法找到最优路线"""
        ga = GeneticAlgorithm(self)
        return ga.optimize()

    def visualize_routes(self, routes: List[List[str]], save_path: str = None):
        """可视化路线"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统可用
        plt.figure(figsize=(15, 10))

        # 绘制所有位置
        for name, loc in self.locations.items():
            x, y = loc.coordinates
            color = 'red' if loc.demand < 0 else 'green'
            size = 100 + abs(loc.demand) * 10  # 根据需求量调整点的大小
            plt.scatter(x, y, c=color, s=size, alpha=0.6)

            # 添加标签
            label = f"{name}\n需求: {loc.demand}\n优先级: {loc.priority}"
            if loc.time_window:
                label += f"\n时间窗口: {loc.time_window[0].strftime('%H:%M')}-{loc.time_window[1].strftime('%H:%M')}"
            plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points')

        # 绘制路线
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            for j in range(len(route) - 1):
                start = self.locations[route[j]].coordinates
                end = self.locations[route[j + 1]].coordinates
                plt.plot([start[0], end[0]], [start[1], end[1]],
                        c=color, linestyle='--', alpha=0.5,
                        label=f'车辆 {i+1}' if j == 0 else "")

        plt.title('共享单车调度路线图')
        plt.xlabel('X 坐标')
        plt.ylabel('Y 坐标')
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @classmethod
    def from_csv(cls, file_path: str, **kwargs) -> 'BicycleScheduler':
        """从CSV文件加载位置数据"""
        locations = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                locations[row['name']] = Location(
                    name=row['name'],
                    coordinates=(float(row['x']), float(row['y'])),
                    demand=int(row['demand']),
                    ideal_count=int(row.get('ideal_count', 0)),
                    current_count=int(row.get('current_count', 0)),
                    priority=int(row.get('priority', 1))
                )
        return cls(locations, **kwargs)
    
    @classmethod
    def from_json(cls, file_path: str, **kwargs) -> 'BicycleScheduler':
        """从JSON文件加载位置数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        locations = {}
        for loc_data in data['locations']:
            locations[loc_data['name']] = Location(**loc_data)
        return cls(locations, **kwargs)

def main():
    # 示例用法
    locations = {
        'A': Location('A', (0, 0), -5, 20, 15, 
                     time_window=(datetime.now(), datetime.now() + timedelta(hours=1)),
                     priority=2),
        'B': Location('B', (4, 3), 10, 15, 25, priority=1),
        'C': Location('C', (2, 6), -6, 18, 12, priority=3),
        'D': Location('D', (5, 5), 6, 22, 28, priority=1),
        'E': Location('E', (7, 6), -9, 16, 7, priority=2),
        'F': Location('F', (3, 2), 4, 19, 23, priority=1),
        'G': Location('G', (1, 1), -2, 17, 15, priority=1)
    }
    
    scheduler = BicycleScheduler(locations)
    optimal_routes = scheduler.find_optimal_routes()
    
    print("最优路线:")
    total_supplied = 0
    total_time = 0
    for i, route in enumerate(optimal_routes):
        time, supplied = scheduler.evaluate_route(route)
        total_supplied += supplied
        total_time += time
        print(f"车辆 {i+1}: {' -> '.join(route)}")
        print(f"时间: {time:.2f} 小时, 供应量: {supplied} 辆自行车")
    
    print(f"\n总供应量: {total_supplied} 辆自行车")
    print(f"总时间: {total_time:.2f} 小时")
    
    scheduler.visualize_routes(optimal_routes, 'scheduling_routes.png')

if __name__ == "__main__":
    main() 