import numpy as np
import json
import pandas as pd
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random
import os  # 添加os模块

@dataclass
class Point:
    name: str
    x: float
    y: float
    bikes: int
    faulty_bikes: int

class MaintenanceOptimizer:
    def __init__(self, parking_points: Dict, distances: Dict, demand_data: pd.DataFrame):
        # 更新维修点坐标
        parking_points["维修点"] = [2200, 600]
        
        # 更新其他点位坐标
        updated_coords = {
            "南门": [1771.3, 3127.6],
            "三食堂": [1474.0, 2520.1],
            "梅苑1栋": [1136.6, 2683.9],
            "校医院": [815.2, 2641.0],
            "二食堂": [843.8, 2353.8],
            "教学4楼": [1755.3, 2223.4],
            "教学2楼": [1481.8, 1747.7],
            "一食堂": [1087.2, 1825.9],
            "菊苑1栋": [961.3, 1490.2],
            "网球场": [1226.7, 1217.2],
            "东门": [2165.2, 1679.7],
            "工程中心": [2008.1, 1007.1],
            "计算机学院": [1539.0, 1089.6],
            "体育馆": [792.8, 970.2],
            "北门": [1502.5, 632.2]
        }
        
        for point in updated_coords:
            if point in parking_points:
                parking_points[point] = updated_coords[point]
        
        # 添加维修点到其他点的距离
        for point in parking_points:
            if point != "维修点":
                dist = np.sqrt((parking_points[point][0] - parking_points["维修点"][0])**2 + 
                              (parking_points[point][1] - parking_points["维修点"][1])**2)
                distances[f"维修点,{point}"] = dist
                distances[f"{point},维修点"] = dist
        
        self.parking_points = parking_points
        self.distances = distances
        self.demand_data = demand_data
        
        # 加载需求预测数据
        self.demand_predictions = {}
        for hour in ['07', '09', '12', '14', '18', '21', '23']:
            try:
                demand_file = f'results/bike_demand_{hour}_00_20250419_144830.csv'
                hour_demand = pd.read_csv(demand_file)
                self.demand_predictions[hour] = hour_demand
            except:
                print(f"Warning: Could not load demand data for {hour}:00")
        
        self.points = self._create_points()
        self.distance_matrix = self._create_distance_matrix()
        self.speed = 25  # 固定速度为25 km/h
        self.load_time = 1/60  # 固定装载时间为1分钟/辆
        self.max_capacity = 20  # 固定容量为20辆
        self.target_fault_rate = 0.06  # 目标故障率6%
        self.max_time = 1.5  # 最大工作时间1.5小时
        
        # 计算并存储初始故障率信息
        self.initial_point_stats = self._calculate_initial_stats()
        
    def _infer_point_data(self, point_name: str, base_demand: pd.DataFrame) -> Tuple[float, float, float]:
        """根据现有数据推断点位的车辆数据"""
        # 计算所有已知点位的平均值和中位数
        avg_bikes = base_demand['现有数量'].mean()
        avg_demand = base_demand['预测需求'].mean()
        avg_ideal = base_demand['理想配置'].mean()
        
        med_bikes = base_demand['现有数量'].median()
        med_demand = base_demand['预测需求'].median()
        med_ideal = base_demand['理想配置'].median()
        
        # 根据点位特征推断数据
        if '食堂' in point_name:
            # 食堂类点位通常需求较大
            similar_points = base_demand[base_demand['点位'].str.contains('食堂')]
            if not similar_points.empty:
                return (
                    similar_points['现有数量'].mean(),
                    similar_points['预测需求'].mean(),
                    similar_points['理想配置'].mean()
                )
        elif '教学' in point_name:
            # 教学楼类点位
            similar_points = base_demand[base_demand['点位'].str.contains('教学')]
            if not similar_points.empty:
                return (
                    similar_points['现有数量'].mean(),
                    similar_points['预测需求'].mean(),
                    similar_points['理想配置'].mean()
                )
        elif any(keyword in point_name for keyword in ['体育馆', '网球场']):
            # 运动场所类点位
            similar_points = base_demand[base_demand['点位'].isin(['体育馆', '网球场'])]
            if not similar_points.empty:
                return (
                    similar_points['现有数量'].mean(),
                    similar_points['预测需求'].mean(),
                    similar_points['理想配置'].mean()
                )
        
        # 如果没有特征匹配，使用整体统计特征
        # 使用中位数和平均数的加权组合以减少极值影响
        return (
            0.7 * med_bikes + 0.3 * avg_bikes,
            0.7 * med_demand + 0.3 * avg_demand,
            0.7 * med_ideal + 0.3 * avg_ideal
        )

    def _create_points(self) -> List[Point]:
        """创建点位列表，考虑预测需求并推断缺失数据"""
        points = []
        total_bikes = 0
        
        # 使用12:00的需求数据作为基准
        base_demand = self.demand_predictions.get('12', pd.DataFrame())
        
        # 确保所有点位都被处理
        all_points = set(self.parking_points.keys()) - {'维修点'}
        processed_points = set()
        
        # 首先添加维修点
        points.append(Point("维修点", self.parking_points["维修点"][0], 
                          self.parking_points["维修点"][1], 0, 0))
        
        # 处理所有其他点位
        for name, coords in self.parking_points.items():
            if name == "维修点":
                continue
                
            processed_points.add(name)
            point_data = base_demand[base_demand['点位'] == name]
            
            if not point_data.empty:
                current_bikes = point_data['现有数量'].values[0]
                predicted_demand = point_data['预测需求'].values[0]
                ideal_config = point_data['理想配置'].values[0]
            else:
                # 推断缺失数据
                print(f"推断点位 {name} 的数据...")
                current_bikes, predicted_demand, ideal_config = self._infer_point_data(name, base_demand)
                print(f"推断结果 - 现有数量: {current_bikes:.1f}, 预测需求: {predicted_demand:.1f}, "
                      f"理想配置: {ideal_config:.1f}")
            
            # 计算故障车辆数，考虑预测需求
            bikes = int(current_bikes)
            if bikes > 0:
                if current_bikes > ideal_config:
                    faulty_bikes = int(min(
                        max(1, (current_bikes - ideal_config) * 0.8),
                        max(1, current_bikes * 0.06)
                    ))
                else:
                    faulty_bikes = int(max(1, current_bikes * 0.06))
            else:
                faulty_bikes = 0
            
            points.append(Point(name, coords[0], coords[1], bikes, faulty_bikes))
            total_bikes += bikes
        
        # 检查是否有遗漏的点位
        missing_points = all_points - processed_points
        if missing_points:
            print(f"\n警告：以下点位在处理过程中被遗漏：{missing_points}")
        
        return points
    
    def _create_distance_matrix(self) -> np.ndarray:
        n = len(self.points)
        matrix = np.zeros((n, n))
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                if i != j:
                    key = f"{p1.name},{p2.name}"
                    # 将距离从米转换为千米
                    matrix[i, j] = self.distances[key] / 1000
        return matrix
    
    def calculate_route_time(self, route: List[int], pickups: List[int]) -> float:
        """计算路线总时间（包括行驶时间和装载时间）"""
        travel_time = 0
        for i in range(len(route) - 1):
            travel_time += self.distance_matrix[route[i], route[i+1]] / self.speed
        load_time = sum(pickups) * self.load_time
        return travel_time + load_time
    
    def calculate_point_fault_rate(self, point_name: str, remaining_faulty: Dict[int, int]) -> float:
        """计算指定点位的当前故障率"""
        point_idx = next(i for i, p in enumerate(self.points) if p.name == point_name)
        point = self.points[point_idx]
        
        if point.bikes == 0:
            return 0
            
        current_faulty = remaining_faulty.get(point_idx, 0)
        return current_faulty / point.bikes
    
    def calculate_overall_fault_rate(self, remaining_faulty: Dict[int, int]) -> float:
        """计算整体故障率"""
        total_bikes = self.initial_point_stats['overall']['total_bikes']
        if total_bikes == 0:
            return 0
            
        current_faulty = sum(remaining_faulty.values())
        return current_faulty / total_bikes
    
    def print_fault_rates(self, remaining_faulty: Dict[int, int]):
        """打印所有点位的故障率信息"""
        print("\n故障率统计：")
        print("点位名称    总车辆数    故障车辆数    初始故障率    当前故障率")
        print("-" * 65)
        
        # 按点位名称排序
        sorted_points = sorted([p for p in self.points if p.name != "维修点"], 
                             key=lambda x: x.name)
        
        for point in sorted_points:
            if point.name != "维修点":
                initial_stats = self.initial_point_stats[point.name]
                current_fault_rate = self.calculate_point_fault_rate(point.name, remaining_faulty)
                print(f"{point.name:<12} {initial_stats['total_bikes']:>8} {initial_stats['faulty_bikes']:>12} "
                      f"{initial_stats['fault_rate']:>12.2%} {current_fault_rate:>12.2%}")
        
        print("-" * 65)
        overall_stats = self.initial_point_stats['overall']
        current_overall_rate = self.calculate_overall_fault_rate(remaining_faulty)
        print(f"总计    {overall_stats['total_bikes']:>8} {overall_stats['total_faulty']:>12} "
              f"{overall_stats['fault_rate']:>12.2%} {current_overall_rate:>12.2%}")
        print(f"目标故障率: {self.target_fault_rate:.2%}")
        print(f"与目标差距: {abs(current_overall_rate - self.target_fault_rate):.2%}")

    def calculate_priority_score(self, point_idx: int, remaining_faulty: Dict[int, int], 
                               current_point: int, capacity: int) -> float:
        """计算点位优先级分数"""
        if point_idx not in remaining_faulty or remaining_faulty[point_idx] == 0:
            return 0
            
        point = self.points[point_idx]
        
        # 计算该点的故障率
        point_fault_rate = self.calculate_point_fault_rate(point.name, remaining_faulty)
        
        # 计算整体故障率
        overall_fault_rate = self.calculate_overall_fault_rate(remaining_faulty)
        
        # 计算如果回收该点的故障车辆后的故障率
        bikes_to_collect = min(remaining_faulty[point_idx], self.max_capacity - capacity)
        future_remaining = remaining_faulty.copy()
        future_remaining[point_idx] -= bikes_to_collect
        future_fault_rate = self.calculate_overall_fault_rate(future_remaining)
        
        # 计算与目标故障率的差距分数
        current_diff = abs(overall_fault_rate - self.target_fault_rate)
        future_diff = abs(future_fault_rate - self.target_fault_rate)
        rate_improvement = max(0, current_diff - future_diff)
        
        # 计算距离分数（距离越近分数越高）
        distance = self.distance_matrix[current_point, point_idx]
        distance_score = 1 / (1 + distance/500)  # 500米作为标准距离
        
        # 计算回收量分数（优先回收数量多的点）
        collection_score = bikes_to_collect / self.max_capacity
        
        # 计算紧急程度分数
        urgency_score = point_fault_rate / self.target_fault_rate
        
        # 计算容量利用率分数
        capacity_score = (capacity + bikes_to_collect) / self.max_capacity
        
        # 综合评分：
        # - 故障率改善权重 0.3
        # - 距离权重 0.2
        # - 回收量权重 0.2
        # - 紧急程度权重 0.2
        # - 容量利用率权重 0.1
        return (0.3 * rate_improvement + 
                0.2 * distance_score + 
                0.2 * collection_score + 
                0.2 * urgency_score + 
                0.1 * capacity_score)
    
    def _calculate_initial_stats(self) -> Dict:
        """计算并返回每个点位的初始统计信息"""
        stats = {}
        total_bikes = 0
        total_faulty = 0
        
        # 按点位名称排序处理所有点位
        sorted_points = sorted([p for p in self.points if p.name != "维修点"], 
                             key=lambda x: x.name)
        
        for point in sorted_points:
            if point.name != "维修点":
                stats[point.name] = {
                    'total_bikes': point.bikes,
                    'faulty_bikes': point.faulty_bikes,
                    'fault_rate': point.faulty_bikes / point.bikes if point.bikes > 0 else 0
                }
                total_bikes += point.bikes
                total_faulty += point.faulty_bikes
        
        stats['overall'] = {
            'total_bikes': total_bikes,
            'total_faulty': total_faulty,
            'fault_rate': total_faulty / total_bikes if total_bikes > 0 else 0
        }
        
        return stats
    
    def ant_colony_optimization(self, n_ants: int = 50, n_iterations: int = 200,
                              alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1):
        """改进的蚁群算法实现，增加局部搜索和自适应参数调整"""
        n_points = len(self.points)
        pheromone = np.ones((n_points, n_points))
        best_trips = []
        best_total_time = float('inf')
        best_fault_rate_diff = float('inf')
        best_total_collected = 0
        maintenance_idx = next(i for i, p in enumerate(self.points) if p.name == "维修点")
        
        # 打印初始信息
        print("\n初始信息：")
        print(f"总点数：{n_points}")
        print("各点故障车辆数和故障率：")
        total_faulty = 0
        points_with_faulty = []
        for i, point in enumerate(self.points):
            if point.name != "维修点":
                fault_rate = point.faulty_bikes / point.bikes if point.bikes > 0 else 0
                print(f"{point.name}: {point.faulty_bikes}辆 (故障率: {fault_rate:.2%})")
                if point.faulty_bikes > 0:
                    points_with_faulty.append((i, point))
                total_faulty += point.faulty_bikes
        
        total_bikes = sum(p.bikes for p in self.points)
        print(f"\n总车辆数：{total_bikes}辆")
        print(f"总故障车辆数：{total_faulty}辆")
        initial_fault_rate = total_faulty / total_bikes
        print(f"初始故障率：{initial_fault_rate:.2%}")
        print(f"目标故障率：{self.target_fault_rate:.2%}")
        
        # 增强的参数自适应机制
        adaptive_params = {
            'alpha': alpha,
            'beta': beta,
            'local_search_prob': 0.1,  # 局部搜索概率
            'diversification_factor': 1.0  # 多样化因子
        }
        
        no_improvement_count = 0
        max_no_improvement = 20
        stable_iterations = 0
        max_stable_iterations = 10
        
        # 解的历史记录
        solution_history = []
        
        # 计算点位聚类
        clusters = self._cluster_points()
        print("\n点位聚类结果：")
        for i, cluster in enumerate(clusters):
            print(f"聚类 {i+1}: {[self.points[idx].name for idx in cluster]}")
        
        def evaluate_solution(trips, remaining):
            """评估解的质量"""
            if not trips:
                return float('inf'), 0, float('inf')
            
            total_time = sum(self.calculate_route_time(route, pickups) for route, pickups in trips)
            if total_time > self.max_time:
                return float('inf'), 0, float('inf')
            
            total_collected = sum(sum(pickups) for _, pickups in trips)
            current_fault_rate = self.calculate_overall_fault_rate(remaining)
            fault_rate_diff = abs(current_fault_rate - self.target_fault_rate)
            
            # 综合评分考虑多个目标
            score = (fault_rate_diff * 1000 +  # 故障率差距
                    (1 - total_collected/total_faulty) * 100 +  # 回收率
                    total_time/self.max_time * 10)  # 时间利用率
            
            return score, total_collected, total_time
        
        def local_search(trips, remaining_faulty):
            """局部搜索优化"""
            improved = False
            best_score, best_collected, best_time = evaluate_solution(trips, remaining_faulty)
            
            # 2-opt局部搜索
            for trip_idx, (route, pickups) in enumerate(trips):
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        # 尝试反转子路径
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_trips = trips.copy()
                        new_trips[trip_idx] = (new_route, pickups)
                        
                        score, collected, time = evaluate_solution(new_trips, remaining_faulty)
                        if score < best_score:
                            trips[:] = new_trips
                            best_score = score
                            improved = True
            
            # 交换相邻簇之间的点
            for i in range(len(trips) - 1):
                route1, pickups1 = trips[i]
                route2, pickups2 = trips[i + 1]
                for j in range(1, len(route1) - 1):
                    for k in range(1, len(route2) - 1):
                        # 尝试交换点
                        new_route1 = route1[:j] + [route2[k]] + route1[j+1:]
                        new_route2 = route2[:k] + [route1[j]] + route2[k+1:]
                        new_trips = trips.copy()
                        new_trips[i] = (new_route1, pickups1)
                        new_trips[i+1] = (new_route2, pickups2)
                        
                        score, collected, time = evaluate_solution(new_trips, remaining_faulty)
                        if score < best_score:
                            trips[:] = new_trips
                            best_score = score
                            improved = True
            
            return improved
        
        def update_adaptive_params(improvement_found: bool):
            """更新自适应参数"""
            if improvement_found:
                adaptive_params['alpha'] = min(2.0, adaptive_params['alpha'] * 1.05)
                adaptive_params['beta'] = max(1.0, adaptive_params['beta'] * 0.95)
                adaptive_params['local_search_prob'] = min(0.3, adaptive_params['local_search_prob'] * 1.1)
                adaptive_params['diversification_factor'] = max(0.5, adaptive_params['diversification_factor'] * 0.9)
            else:
                adaptive_params['alpha'] = max(0.5, adaptive_params['alpha'] * 0.95)
                adaptive_params['beta'] = min(3.0, adaptive_params['beta'] * 1.05)
                adaptive_params['local_search_prob'] = max(0.05, adaptive_params['local_search_prob'] * 0.9)
                adaptive_params['diversification_factor'] = min(2.0, adaptive_params['diversification_factor'] * 1.1)
        
        for iteration in range(n_iterations):
            iteration_best_score = float('inf')
            iteration_best_trips = None
            iteration_best_remaining = None
            
            # 动态调整蚂蚁数量
            current_ants = int(n_ants * adaptive_params['diversification_factor'])
            
            for ant in range(current_ants):
                remaining_faulty = {i: p.faulty_bikes for i, p in enumerate(self.points) 
                                  if p.name != "维修点" and p.faulty_bikes > 0}
                trips = []
                total_time = 0
                
                # 优先处理故障车辆多的簇
                sorted_clusters = sorted(clusters, 
                    key=lambda c: sum(self.points[i].faulty_bikes for i in c), 
                    reverse=True)
                
                for cluster in sorted_clusters:
                    cluster_remaining = {i: remaining_faulty[i] 
                                      for i in remaining_faulty if i in cluster}
                    
                    while cluster_remaining and total_time < self.max_time:
                        current_point = maintenance_idx
                        route = [current_point]
                        pickups = [0]
                        capacity = 0
                        
                        # 计算最佳起始点
                        start_candidates = []
                        for point in cluster_remaining:
                            score = (cluster_remaining[point] / self.max_capacity) * \
                                   (1 / (1 + self.distance_matrix[current_point, point]))
                            start_candidates.append((point, score))
                        
                        if start_candidates:
                            start_candidates.sort(key=lambda x: x[1], reverse=True)
                            next_point = start_candidates[0][0]
                            route.append(next_point)
                            pickup_amount = min(cluster_remaining[next_point], 
                                             self.max_capacity)
                            pickups.append(pickup_amount)
                            capacity += pickup_amount
                            cluster_remaining[next_point] -= pickup_amount
                            remaining_faulty[next_point] -= pickup_amount
                            if cluster_remaining[next_point] == 0:
                                del cluster_remaining[next_point]
                            if remaining_faulty[next_point] == 0:
                                del remaining_faulty[next_point]
                            current_point = next_point
                        
                        while cluster_remaining and capacity < self.max_capacity:
                            candidates = []
                            for next_point in cluster_remaining:
                                if capacity + cluster_remaining[next_point] <= self.max_capacity:
                                    priority_score = self.calculate_priority_score(
                                        next_point, remaining_faulty, current_point, capacity)
                                    tau = pheromone[current_point, next_point]
                                    eta = 1 / (self.distance_matrix[current_point, next_point] + 0.1)
                                    score = (tau ** adaptive_params['alpha']) * \
                                           (eta ** adaptive_params['beta']) * \
                                           (priority_score ** 1.5)  # 增加优先级权重
                                    candidates.append((next_point, score))
                            
                            if not candidates:
                                break
                            
                            # 改进的选择机制
                            total_score = sum(score for _, score in candidates)
                            if total_score == 0:
                                break
                            
                            # 轮盘赌选择
                            probabilities = [score/total_score for _, score in candidates]
                            next_point = np.random.choice([p for p, _ in candidates], p=probabilities)
                            
                            route.append(next_point)
                            pickup_amount = min(cluster_remaining[next_point], 
                                             self.max_capacity - capacity)
                            pickups.append(pickup_amount)
                            capacity += pickup_amount
                            
                            cluster_remaining[next_point] -= pickup_amount
                            remaining_faulty[next_point] -= pickup_amount
                            if cluster_remaining[next_point] == 0:
                                del cluster_remaining[next_point]
                            if remaining_faulty[next_point] == 0:
                                del remaining_faulty[next_point]
                            
                            current_point = next_point
                        
                        # 返回维修点
                        route.append(maintenance_idx)
                        pickups.append(0)
                        
                        # 计算行程时间
                        trip_time = self.calculate_route_time(route, pickups)
                        if total_time + trip_time > self.max_time:
                            break
                        
                        total_time += trip_time
                        trips.append((route, pickups))
                
                # 评估当前解
                score, total_collected, total_time = evaluate_solution(trips, remaining_faulty)
                
                # 更新迭代最佳解
                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_trips = trips.copy()
                    iteration_best_remaining = remaining_faulty.copy()
            
            # 对迭代最佳解进行局部搜索
            if iteration_best_trips and random.random() < adaptive_params['local_search_prob']:
                improved = local_search(iteration_best_trips, iteration_best_remaining)
                if improved:
                    score, total_collected, total_time = evaluate_solution(
                        iteration_best_trips, iteration_best_remaining)
                    iteration_best_score = score
            
            # 更新全局最佳解
            if iteration_best_score < float('inf'):
                score, total_collected, total_time = evaluate_solution(
                    iteration_best_trips, iteration_best_remaining)
                
                if (score < best_fault_rate_diff or
                    (score == best_fault_rate_diff and total_collected > best_total_collected) or
                    (score == best_fault_rate_diff and total_collected == best_total_collected and 
                     total_time < best_total_time)):
                    best_fault_rate_diff = score
                    best_total_collected = total_collected
                    best_total_time = total_time
                    best_trips = iteration_best_trips.copy()
                    no_improvement_count = 0
                    stable_iterations = 0
                    
                    print(f"\n找到新解 (迭代 {iteration + 1}):")
                    print(f"总回收车辆数: {best_total_collected}")
                    print(f"总行程数: {len(best_trips)}")
                    print(f"总时间: {best_total_time:.2f}小时")
                    current_fault_rate = self.calculate_overall_fault_rate(iteration_best_remaining)
                    print(f"当前故障率: {current_fault_rate:.2%}")
                    print(f"与目标故障率差距: {abs(current_fault_rate - self.target_fault_rate):.2%}")
                    
                    # 更新自适应参数
                    update_adaptive_params(True)
                else:
                    no_improvement_count += 1
                    stable_iterations += 1
                    update_adaptive_params(False)
            
            # 更新信息素
            pheromone *= (1 - rho)
            if best_trips:
                for route, pickups in best_trips:
                    quality = 1 / (best_fault_rate_diff + 0.1)
                    for i in range(len(route) - 1):
                        pheromone[route[i], route[i+1]] += quality * \
                            (sum(pickups) / best_total_time) * \
                            adaptive_params['diversification_factor']
            
            # 信息素范围限制
            pheromone = np.clip(pheromone, 0.1, 2.0)
            
            # 检查是否达到稳定状态
            if stable_iterations >= max_stable_iterations:
                print("\n达到稳定状态，输出最终结果：")
                if best_trips:
                    final_fault_rate = (total_faulty - best_total_collected) / total_bikes
                    print(f"总回收车辆数：{best_total_collected}辆")
                    print(f"总时间：{best_total_time:.2f}小时")
                    print(f"最终故障率：{final_fault_rate:.2%}")
                    print(f"与目标故障率差距：{abs(final_fault_rate - self.target_fault_rate):.2%}")
                return best_trips
        
        return best_trips
    
    def _cluster_points(self) -> List[List[int]]:
        """将点位按地理位置和故障车辆数量聚类"""
        # 排除维修点
        points_coords = [(i, [p.x, p.y], p.faulty_bikes) 
                        for i, p in enumerate(self.points) if p.name != "维修点"]
        point_indices = [i for i, _, _ in points_coords]
        
        # 使用基于距离和故障车辆数量的聚类
        clusters = []
        unassigned = set(point_indices)
        
        # 首先处理故障车辆数量较多的点
        sorted_points = sorted(points_coords, key=lambda x: x[2], reverse=True)
        
        while unassigned:
            # 选择故障车辆最多的未分配点作为中心
            center = next(p[0] for p in sorted_points if p[0] in unassigned)
            unassigned.remove(center)
            cluster = {center}
            
            # 计算到中心点的距离和故障车辆数量的综合得分
            scores = []
            for point in list(unassigned):
                distance = self.distance_matrix[center, point]
                faulty_bikes = self.points[point].faulty_bikes
                # 综合得分 = 距离得分 + 故障车辆数量得分
                score = (1000 - min(distance, 1000))/1000 + faulty_bikes/10
                scores.append((point, score))
            
            # 按得分排序
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # 选择得分最高的几个点加入簇
            for point, score in scores:
                if point in unassigned and len(cluster) < 5:  # 限制每个簇的大小
                    cluster.add(point)
                    unassigned.remove(point)
            
            clusters.append(list(cluster))
        
        return clusters
    
    def plot_trips(self, trips: List[Tuple[List[int], List[int]]], save_path: str = None):
        """绘制所有行程的路线图"""
        plt.figure(figsize=(12, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 获取y轴最大值
        max_y = max(point.y for point in self.points)
        
        # 绘制所有点
        for point in self.points:
            if point.faulty_bikes > 0:
                plt.plot(point.x, max_y - point.y, 'bo')  # 反转y坐标
                plt.text(point.x, max_y - point.y + 20, f"{point.name}\n({point.faulty_bikes})", fontsize=8)
            elif point.name == "维修点":
                plt.plot(point.x, max_y - point.y, 'r*', markersize=12)  # 反转y坐标
                plt.text(point.x, max_y - point.y + 20, point.name, fontsize=8)
        
        # 为每个行程使用不同的颜色
        colors = ['r-', 'g-', 'b-', 'c-', 'm-', 'y-']
        
        # 绘制路线
        for i, (route, pickups) in enumerate(trips):
            color = colors[i % len(colors)]
            for j in range(len(route) - 1):
                p1 = self.points[route[j]]
                p2 = self.points[route[j+1]]
                plt.plot([p1.x, p2.x], [max_y - p1.y, max_y - p2.y], color)  # 反转y坐标
                if pickups[j+1] > 0:
                    plt.text((p1.x + p2.x)/2, (max_y - p1.y + max_y - p2.y)/2, 
                            f"{pickups[j+1]}", color='black', fontsize=8)
        
        plt.title("故障车辆回收路线优化（多次往返）")
        plt.grid(True)
        
        # 如果提供了保存路径，则保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n路线图已保存至: {save_path}")
        
        plt.show()

def main():
    # 加载数据
    with open('processed_data/processed_parking_points.json', 'r', encoding='utf-8') as f:
        parking_points = json.load(f)
    
    with open('processed_data/processed_distances.json', 'r', encoding='utf-8') as f:
        distances = json.load(f)
    
    demand_data = pd.read_csv('processed_data/processed_demand.csv')
    
    # 创建优化器
    optimizer = MaintenanceOptimizer(parking_points, distances, demand_data)
    
    # 运行优化
    best_trips = optimizer.ant_colony_optimization()
    
    # 输出结果
    print("\n优化结果：")
    if not best_trips:
        print("未找到有效解")
        print("建议：")
        print("1. 增加运输车辆数量")
        print("2. 减少单次回收的故障车辆数量")
    else:
        total_picked = sum(sum(pickups) for _, pickups in best_trips)
        total_time = sum(optimizer.calculate_route_time(route, pickups) for route, pickups in best_trips)
        print(f"\n总回收车辆数：{total_picked}辆")
        print(f"总行程数：{len(best_trips)}次")
        print(f"预计总时间：{total_time:.2f}小时")
        
        # 打印每个行程的详细信息
        for i, (route, pickups) in enumerate(best_trips, 1):
            print(f"\n行程 {i}:")
            print(f"回收车辆数: {sum(pickups)}")
            print(f"行程时间: {optimizer.calculate_route_time(route, pickups):.2f}小时")
            print("路线详情:")
            for j in range(len(route) - 1):
                p1 = optimizer.points[route[j]]
                p2 = optimizer.points[route[j+1]]
                print(f"{p1.name} -> {p2.name} (回收{pickups[j+1]}辆)")
        
        # 创建output目录（如果不存在）
        os.makedirs('output', exist_ok=True)
        
        # 保存并显示路线图
        save_path = 'output/maintenance_route.png'
        optimizer.plot_trips(best_trips, save_path)
        
        # 保存结果到文件
        result_data = {
            'total_picked': total_picked,
            'total_trips': len(best_trips),
            'total_time': total_time,
            'trips': []
        }
        
        for i, (route, pickups) in enumerate(best_trips, 1):
            trip_info = {
                'trip_number': i,
                'picked_bikes': sum(pickups),
                'time': optimizer.calculate_route_time(route, pickups),
                'route': []
            }
            
            for j in range(len(route) - 1):
                p1 = optimizer.points[route[j]]
                p2 = optimizer.points[route[j+1]]
                trip_info['route'].append({
                    'from': p1.name,
                    'to': p2.name,
                    'picked': pickups[j+1]
                })
            
            result_data['trips'].append(trip_info)
        
        # 保存结果到JSON文件
        with open('results/maintenance_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存至: results/maintenance_result.json")

if __name__ == "__main__":
    main() 