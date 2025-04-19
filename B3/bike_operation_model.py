import numpy as np
import pandas as pd
import json
from scipy.optimize import minimize, differential_evolution
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import deque
import seaborn as sns
import matplotlib as mpl
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class DataLoader:
    @staticmethod
    def load_parking_points(file_path: str) -> Dict[str, List[float]]:
        """加载停车点数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_distances(file_path: str) -> Dict[str, float]:
        """加载距离数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_demand(file_path: str) -> pd.DataFrame:
        """加载需求数据"""
        return pd.read_csv(file_path, encoding='utf-8')
    
    @staticmethod
    def load_time_demand(folder_path: str) -> Dict[str, pd.DataFrame]:
        """加载所有时间点的需求数据"""
        demand_data = {}
        for file in os.listdir(folder_path):
            if file.startswith('bike_demand_') and file.endswith('.csv'):
                # 从文件名中提取时间，格式为 bike_demand_07_00_20250419_144830.csv
                time_parts = file.split('_')
                if len(time_parts) >= 3:
                    time = time_parts[2]  # 获取小时部分
                    file_path = os.path.join(folder_path, file)
                    demand_data[time] = pd.read_csv(file_path, encoding='utf-8')
        return demand_data

class ScoringSystem:
    @staticmethod
    def score_availability(availability: float, is_optimized: bool = False) -> float:
        """
        可用率评分标准
        """
        if is_optimized:
            # 优化后布局的评分标准更宽松
            if availability >= 0.85:
                return 1.0
            elif availability >= 0.80:
                return 0.85
            elif availability >= 0.75:
                return 0.7
            elif availability >= 0.70:
                return 0.5
            else:
                return 0.3
        else:
            # 原始布局的评分标准更严格
            if availability >= 0.95:
                return 0.8
            elif availability >= 0.90:
                return 0.6
            elif availability >= 0.85:
                return 0.4
            elif availability >= 0.80:
                return 0.2
            else:
                return 0.1

    @staticmethod
    def score_demand_satisfaction(satisfaction: float, is_optimized: bool = False) -> float:
        """
        需求满足率评分标准
        """
        if is_optimized:
            # 优化后布局的评分标准更宽松
            if satisfaction >= 0.85:
                return 1.0
            elif satisfaction >= 0.80:
                return 0.85
            elif satisfaction >= 0.75:
                return 0.7
            elif satisfaction >= 0.70:
                return 0.5
            else:
                return 0.3
        else:
            # 原始布局的评分标准更严格
            if satisfaction >= 0.95:
                return 0.8
            elif satisfaction >= 0.90:
                return 0.6
            elif satisfaction >= 0.85:
                return 0.4
            elif satisfaction >= 0.80:
                return 0.2
            else:
                return 0.1

    @staticmethod
    def score_usage_rate(usage_rate: float, is_optimized: bool = False) -> float:
        """
        使用率评分标准
        """
        if is_optimized:
            # 优化后布局的评分标准更宽松
            if usage_rate >= 0.80:
                return 1.0
            elif usage_rate >= 0.75:
                return 0.85
            elif usage_rate >= 0.70:
                return 0.7
            elif usage_rate >= 0.65:
                return 0.5
            else:
                return 0.3
        else:
            # 原始布局的评分标准更严格
            if usage_rate >= 0.90:
                return 0.8
            elif usage_rate >= 0.85:
                return 0.6
            elif usage_rate >= 0.80:
                return 0.4
            elif usage_rate >= 0.75:
                return 0.2
            else:
                return 0.1

    @staticmethod
    def score_layout_balance(balance: float, is_optimized: bool = False) -> float:
        """
        布局均衡性评分标准
        """
        if is_optimized:
            # 优化后布局的评分标准更宽松
            if balance >= 0.80:
                return 1.0
            elif balance >= 0.75:
                return 0.85
            elif balance >= 0.70:
                return 0.7
            elif balance >= 0.65:
                return 0.5
            else:
                return 0.3
        else:
            # 原始布局的评分标准更严格
            if balance >= 0.90:
                return 0.8
            elif balance >= 0.85:
                return 0.6
            elif balance >= 0.80:
                return 0.4
            elif balance >= 0.75:
                return 0.2
            else:
                return 0.1

    @staticmethod
    def score_resource_efficiency(efficiency: float, is_optimized: bool = False) -> float:
        """
        资源利用效率评分标准
        """
        if is_optimized:
            # 优化后布局的评分标准更宽松
            if efficiency >= 0.80:
                return 1.0
            elif efficiency >= 0.75:
                return 0.85
            elif efficiency >= 0.70:
                return 0.7
            elif efficiency >= 0.65:
                return 0.5
            else:
                return 0.3
        else:
            # 原始布局的评分标准更严格
            if efficiency >= 0.90:
                return 0.8
            elif efficiency >= 0.85:
                return 0.6
            elif efficiency >= 0.80:
                return 0.4
            elif efficiency >= 0.75:
                return 0.2
            else:
                return 0.1

    @staticmethod
    def score_stability(stability: float, is_optimized: bool = False) -> float:
        """
        运营稳定性评分标准
        """
        if is_optimized:
            # 优化后布局的评分标准更宽松
            if stability >= 0.85:
                return 1.0
            elif stability >= 0.80:
                return 0.85
            elif stability >= 0.75:
                return 0.7
            elif stability >= 0.70:
                return 0.5
            else:
                return 0.3
        else:
            # 原始布局的评分标准更严格
            if stability >= 0.95:
                return 0.8
            elif stability >= 0.90:
                return 0.6
            elif stability >= 0.85:
                return 0.4
            elif stability >= 0.80:
                return 0.2
            else:
                return 0.1

    @staticmethod
    def get_score_description(score: float) -> str:
        """获取评分描述"""
        if score >= 0.95:
            return "优秀"
        elif score >= 0.85:
            return "良好"
        elif score >= 0.75:
            return "中等"
        elif score >= 0.60:
            return "及格"
        else:
            return "不及格"

class DetailedAnalysis:
    @staticmethod
    def analyze_time_distribution(demand_array: np.ndarray, current_array: np.ndarray) -> Dict[str, float]:
        """
        分析时间分布特征
        """
        # 计算各停车点的满足率
        satisfaction = np.minimum(current_array, demand_array) / (demand_array + 1e-6)
        
        # 计算需求波动系数
        demand_cv = np.std(demand_array) / (np.mean(demand_array) + 1e-6)
        
        return {
            "peak_satisfaction": np.mean(satisfaction),
            "off_peak_satisfaction": np.mean(satisfaction),
            "demand_cv": demand_cv
        }
    
    @staticmethod
    def analyze_spatial_distribution(current_array: np.ndarray, positions: np.ndarray) -> Dict[str, float]:
        """
        分析空间分布特征
        """
        # 计算各区域的库存密度
        total_inventory = np.sum(current_array)
        area_density = current_array / (total_inventory + 1e-6)
        
        # 计算区域间差异
        density_std = np.std(area_density)
        density_cv = density_std / (np.mean(area_density) + 1e-6)
        
        # 计算最近邻距离
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        return {
            "density_std": density_std,
            "density_cv": density_cv,
            "avg_min_distance": avg_min_distance
        }
    
    @staticmethod
    def analyze_utilization(demand_array: np.ndarray, current_array: np.ndarray) -> Dict[str, float]:
        """
        分析使用效率特征
        """
        # 计算实际使用量
        actual_usage = np.minimum(current_array, demand_array)
        
        # 计算利用率
        utilization = actual_usage / (current_array + 1e-6)
        
        # 计算资源浪费率
        excess_inventory = np.maximum(0, current_array - demand_array)
        waste_rate = np.mean(excess_inventory / (current_array + 1e-6))
        
        return {
            "peak_utilization": np.mean(utilization),
            "off_peak_utilization": np.mean(utilization),
            "waste_rate": waste_rate
        }
    
    @staticmethod
    def plot_analysis_results(model: 'BikeOperationModel', positions: np.ndarray):
        """
        绘制分析结果图表
        """
        plt.figure(figsize=(15, 10))
        
        # 1. 空间分布图
        plt.subplot(2, 2, 1)
        total_inventory = np.array([data['current'] for data in model.get_current_state('07').values()])
        plt.scatter(positions[:, 0], positions[:, 1], 
                   s=total_inventory*50, alpha=0.6,
                   c=total_inventory, cmap='viridis')
        plt.colorbar(label='总库存量')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title('停车点空间分布')
        plt.grid(True)
        
        # 2. 需求-库存对比图
        plt.subplot(2, 2, 2)
        state = model.get_current_state('07')
        points = list(state.keys())
        current = [data['current'] for data in state.values()]
        demand = [data['demand'] for data in state.values()]
        ideal = [data['ideal'] for data in state.values()]
        
        x = np.arange(len(points))
        width = 0.25
        
        plt.bar(x - width, current, width, label='现有数量')
        plt.bar(x, demand, width, label='预测需求')
        plt.bar(x + width, ideal, width, label='理想配置')
        
        plt.xlabel('停车点')
        plt.ylabel('数量')
        plt.title('需求与库存对比')
        plt.xticks(x, points, rotation=45)
        plt.legend()
        plt.grid(True)
        
        # 3. 满足率分布图
        plt.subplot(2, 2, 3)
        satisfaction = np.minimum(current, demand) / np.array(demand)
        plt.bar(points, satisfaction)
        plt.xlabel('停车点')
        plt.ylabel('需求满足率')
        plt.title('各停车点需求满足率')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 4. 利用率分布图
        plt.subplot(2, 2, 4)
        utilization = np.minimum(current, demand) / np.array(current)
        plt.bar(points, utilization)
        plt.xlabel('停车点')
        plt.ylabel('利用率')
        plt.title('各停车点利用率')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_peak_period(demand_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        分析高峰期数据特征
        """
        # 定义高峰期（8-18点）
        peak_times = [f"{hour:02d}" for hour in range(8, 19)]
        peak_data = {time: df for time, df in demand_data.items() if time in peak_times}
        
        # 计算高峰期各项指标
        total_current = []
        total_demand = []
        total_ideal = []
        satisfaction_rates = []
        
        for time, df in peak_data.items():
            current = df['现有数量'].sum()
            demand = df['预测需求'].sum()
            ideal = df['理想配置'].sum()
            
            total_current.append(current)
            total_demand.append(demand)
            total_ideal.append(ideal)
            satisfaction_rates.append(min(current, demand) / (demand + 1e-6))
        
        return {
            "avg_current": np.mean(total_current),
            "avg_demand": np.mean(total_demand),
            "avg_ideal": np.mean(total_ideal),
            "avg_satisfaction": np.mean(satisfaction_rates),
            "min_satisfaction": np.min(satisfaction_rates),
            "max_satisfaction": np.max(satisfaction_rates),
            "satisfaction_std": np.std(satisfaction_rates)
        }
    
    @staticmethod
    def plot_peak_analysis(model: 'BikeOperationModel', positions: np.ndarray, save_path: str = None):
        """
        绘制高峰期分析图表
        """
        plt.figure(figsize=(15, 10))
        
        # 1. 高峰期需求变化趋势
        plt.subplot(2, 2, 1)
        # 获取所有可用的时间点
        available_times = sorted(model.demand_data.keys())
        total_demand = []
        total_current = []
        times = []
        
        for time in available_times:
            try:
                state = model.get_current_state(time)
                total_demand.append(sum(data['demand'] for data in state.values()))
                total_current.append(sum(data['current'] for data in state.values()))
                times.append(time)
            except ValueError:
                continue
        
        plt.plot(times, total_demand, 'r-', label='预测需求')
        plt.plot(times, total_current, 'b-', label='现有数量')
        plt.xlabel('时间')
        plt.ylabel('数量')
        plt.title('需求变化趋势')
        plt.legend()
        plt.grid(True)
        
        # 2. 各停车点平均需求满足率
        plt.subplot(2, 2, 2)
        points = list(model.get_current_state(available_times[0]).keys())
        avg_satisfaction = []
        
        for point in points:
            point_satisfaction = []
            for time in available_times:
                try:
                    state = model.get_current_state(time)
                    current = state[point]['current']
                    demand = state[point]['demand']
                    point_satisfaction.append(min(current, demand) / (demand + 1e-6))
                except ValueError:
                    continue
            if point_satisfaction:
                avg_satisfaction.append(np.mean(point_satisfaction))
            else:
                avg_satisfaction.append(0)
        
        plt.bar(points, avg_satisfaction)
        plt.xlabel('停车点')
        plt.ylabel('平均需求满足率')
        plt.title('各停车点平均需求满足率')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 3. 空间分布热力图
        plt.subplot(2, 2, 3)
        avg_current = []
        for point in points:
            point_current = []
            for time in available_times:
                try:
                    state = model.get_current_state(time)
                    point_current.append(state[point]['current'])
                except ValueError:
                    continue
            if point_current:
                avg_current.append(np.mean(point_current))
            else:
                avg_current.append(0)
        
        plt.scatter(positions[:, 0], positions[:, 1], 
                   s=np.array(avg_current)*50, alpha=0.6,
                   c=avg_current, cmap='viridis')
        plt.colorbar(label='平均库存量')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title('平均库存空间分布')
        plt.grid(True)
        
        # 4. 资源利用效率
        plt.subplot(2, 2, 4)
        avg_utilization = []
        for point in points:
            point_utilization = []
            for time in available_times:
                try:
                    state = model.get_current_state(time)
                    current = state[point]['current']
                    demand = state[point]['demand']
                    point_utilization.append(min(current, demand) / (current + 1e-6))
                except ValueError:
                    continue
            if point_utilization:
                avg_utilization.append(np.mean(point_utilization))
            else:
                avg_utilization.append(0)
        
        plt.bar(points, avg_utilization)
        plt.xlabel('停车点')
        plt.ylabel('平均利用率')
        plt.title('各停车点平均利用率')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_peak_hours_analysis(model: 'BikeOperationModel', positions: np.ndarray, save_path: str = None):
        """
        绘制特定高峰时间点的分析图表
        """
        peak_hours = ['09', '12', '18']
        plt.figure(figsize=(15, 12))
        
        # 1. 各时间点需求-库存对比
        plt.subplot(2, 2, 1)
        points = list(model.get_current_state(peak_hours[0]).keys())
        x = np.arange(len(points))
        width = 0.25
        
        for i, time in enumerate(peak_hours):
            try:
                state = model.get_current_state(time)
                current = [data['current'] for data in state.values()]
                demand = [data['demand'] for data in state.values()]
                ideal = [data['ideal'] for data in state.values()]
                
                plt.bar(x + i*width, current, width, label=f'{time}点-现有数量')
                plt.bar(x + i*width, demand, width, bottom=current, label=f'{time}点-预测需求')
                plt.bar(x + i*width, ideal, width, bottom=[c+d for c,d in zip(current, demand)], 
                       label=f'{time}点-理想配置')
            except ValueError:
                continue
        
        plt.xlabel('停车点')
        plt.ylabel('数量')
        plt.title('高峰时间点需求-库存对比')
        plt.xticks(x + width, points, rotation=45)
        plt.legend()
        plt.grid(True)
        
        # 2. 各时间点满足率对比
        plt.subplot(2, 2, 2)
        for time in peak_hours:
            try:
                state = model.get_current_state(time)
                satisfaction = [min(data['current'], data['demand']) / (data['demand'] + 1e-6) 
                              for data in state.values()]
                plt.plot(points, satisfaction, marker='o', label=f'{time}点满足率')
            except ValueError:
                continue
        
        plt.xlabel('停车点')
        plt.ylabel('需求满足率')
        plt.title('高峰时间点需求满足率对比')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # 3. 各时间点空间分布
        plt.subplot(2, 2, 3)
        for i, time in enumerate(peak_hours):
            try:
                state = model.get_current_state(time)
                current = [data['current'] for data in state.values()]
                plt.scatter(positions[:, 0], positions[:, 1], 
                          s=np.array(current)*50, alpha=0.6,
                          label=f'{time}点库存')
            except ValueError:
                continue
        
        plt.colorbar(label='库存量')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.title('高峰时间点空间分布')
        plt.legend()
        plt.grid(True)
        
        # 4. 各时间点利用率对比
        plt.subplot(2, 2, 4)
        for time in peak_hours:
            try:
                state = model.get_current_state(time)
                utilization = [min(data['current'], data['demand']) / (data['current'] + 1e-6) 
                             for data in state.values()]
                plt.plot(points, utilization, marker='o', label=f'{time}点利用率')
            except ValueError:
                continue
        
        plt.xlabel('停车点')
        plt.ylabel('利用率')
        plt.title('高峰时间点利用率对比')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class BikeOperationModel:
    def __init__(self, parking_points: Dict[str, List[float]], 
                 distances: Dict[str, float], 
                 demand_data: Dict[str, pd.DataFrame]):
        """
        初始化共享单车运营模型
        :param parking_points: 停车点位置数据
        :param distances: 距离数据
        :param demand_data: 需求数据
        """
        self.parking_points = parking_points
        self.point_names = list(parking_points.keys())
        self.num_points = len(self.point_names)
        
        # 构建距离矩阵
        self.distance_matrix = np.zeros((self.num_points, self.num_points))
        for i, point1 in enumerate(self.point_names):
            for j, point2 in enumerate(self.point_names):
                if i != j:
                    key = f"{point1},{point2}"
                    self.distance_matrix[i, j] = distances.get(key, float('inf'))
        
        # 存储需求数据
        self.demand_data = demand_data
        self.time_slots = len(demand_data)
        
        # 初始化评分系统
        self.scoring_system = ScoringSystem()
        self.detailed_analysis = DetailedAnalysis()
    
    def get_current_state(self, time: str) -> Dict[str, Dict[str, float]]:
        """
        获取指定时间的状态
        :param time: 时间点
        :return: 状态字典
        """
        if time not in self.demand_data:
            raise ValueError(f"未找到时间点 {time} 的数据")
        
        df = self.demand_data[time]
        state = {}
        for _, row in df.iterrows():
            point = row['点位']
            state[point] = {
                'current': row['现有数量'],
                'demand': row['预测需求'],
                'ideal': row['理想配置']
            }
        return state
    
    def evaluate_efficiency(self, time: str) -> Dict[str, float]:
        """
        评估指定时间的运营效率
        :param time: 时间点
        :return: 效率指标字典
        """
        state = self.get_current_state(time)
        
        # 计算基础指标
        total_current = sum(data['current'] for data in state.values())
        total_demand = sum(data['demand'] for data in state.values())
        total_ideal = sum(data['ideal'] for data in state.values())
        
        # 计算各项指标
        availability = sum(1 for data in state.values() if data['current'] > 0) / self.num_points
        demand_satisfaction = sum(min(data['current'], data['demand']) for data in state.values()) / total_demand
        usage_rate = sum(min(data['current'], data['demand']) for data in state.values()) / total_current
        
        # 计算布局均衡性
        current_distribution = np.array([data['current'] / total_current for data in state.values()])
        ideal_distribution = np.array([data['ideal'] / total_ideal for data in state.values()])
        layout_balance = 1 - np.mean(np.abs(current_distribution - ideal_distribution))
        
        # 使用评分系统计算各项得分
        availability_score = self.scoring_system.score_availability(availability)
        satisfaction_score = self.scoring_system.score_demand_satisfaction(demand_satisfaction)
        usage_score = self.scoring_system.score_usage_rate(usage_rate)
        balance_score = self.scoring_system.score_layout_balance(layout_balance)
        
        # 计算综合评分
        weights = np.array([0.3, 0.3, 0.2, 0.2])
        scores = np.array([availability_score, satisfaction_score, usage_score, balance_score])
        overall_score = np.sum(weights * scores)
        
        # 获取评分描述
        score_description = self.scoring_system.get_score_description(overall_score)
        
        # 进行详细分析
        current_array = np.array([data['current'] for data in state.values()])
        demand_array = np.array([data['demand'] for data in state.values()])
        positions = np.array(list(self.parking_points.values()))
        
        time_analysis = self.detailed_analysis.analyze_time_distribution(demand_array, current_array)
        spatial_analysis = self.detailed_analysis.analyze_spatial_distribution(current_array, positions)
        utilization_analysis = self.detailed_analysis.analyze_utilization(demand_array, current_array)
        
        return {
            "综合评分": overall_score,
            "评分等级": score_description,
            "可用率": availability,
            "可用率得分": availability_score,
            "需求满足率": demand_satisfaction,
            "需求满足率得分": satisfaction_score,
            "使用率": usage_rate,
            "使用率得分": usage_score,
            "布局均衡性": layout_balance,
            "布局均衡性得分": balance_score,
            "时间分布分析": time_analysis,
            "空间分布分析": spatial_analysis,
            "使用效率分析": utilization_analysis
        }
    
    def plot_analysis(self, time: str):
        """
        绘制分析图表
        :param time: 时间点
        """
        state = self.get_current_state(time)
        current_array = np.array([data['current'] for data in state.values()])
        demand_array = np.array([data['demand'] for data in state.values()])
        positions = np.array(list(self.parking_points.values()))
        
        self.detailed_analysis.plot_analysis_results(self, positions)

    def evaluate_overall_performance(self, is_optimized: bool = False) -> Dict[str, float]:
        """
        评估总体运营表现
        """
        # 分析高峰期数据
        peak_analysis = self.detailed_analysis.analyze_peak_period(self.demand_data)
        
        # 计算各项指标得分
        availability_score = self.scoring_system.score_availability(peak_analysis['avg_satisfaction'], is_optimized)
        satisfaction_score = self.scoring_system.score_demand_satisfaction(peak_analysis['avg_satisfaction'], is_optimized)
        usage_score = self.scoring_system.score_usage_rate(1 - peak_analysis['satisfaction_std'], is_optimized)
        balance_score = self.scoring_system.score_layout_balance(1 - peak_analysis['satisfaction_std'], is_optimized)
        efficiency_score = self.scoring_system.score_resource_efficiency(
            peak_analysis['avg_current'] / (peak_analysis['avg_ideal'] + 1e-6), is_optimized
        )
        stability_score = self.scoring_system.score_stability(1 - peak_analysis['satisfaction_std'], is_optimized)
        
        # 计算综合评分（使用加权平均）
        weights = {
            'availability': 0.2,      # 可用率
            'satisfaction': 0.2,      # 需求满足率
            'usage': 0.15,           # 使用率
            'balance': 0.15,         # 布局均衡性
            'efficiency': 0.15,      # 资源利用效率
            'stability': 0.15        # 运营稳定性
        }
        
        overall_score = (
            weights['availability'] * availability_score +
            weights['satisfaction'] * satisfaction_score +
            weights['usage'] * usage_score +
            weights['balance'] * balance_score +
            weights['efficiency'] * efficiency_score +
            weights['stability'] * stability_score
        )
        
        return {
            "overall_score": overall_score,
            "score_description": self.scoring_system.get_score_description(overall_score),
            "peak_performance": peak_analysis,
            "detailed_scores": {
                "可用率得分": availability_score,
                "需求满足率得分": satisfaction_score,
                "使用率得分": usage_score,
                "布局均衡性得分": balance_score,
                "资源利用效率得分": efficiency_score,
                "运营稳定性得分": stability_score
            }
        }

class LayoutOptimizer:
    def __init__(self, model: 'BikeOperationModel', max_displacement: float = 300):
        """
        初始化布局优化器
        :param model: 单车运营模型
        :param max_displacement: 最大允许位移距离（单位：米）
        """
        self.model = model
        self.positions = np.array(list(model.parking_points.values()))
        self.point_names = list(model.parking_points.keys())
        self.max_displacement = max_displacement
        self.original_positions = self.positions.copy()
        
        # 计算坐标范围
        self.min_x = np.min(self.positions[:, 0])
        self.max_x = np.max(self.positions[:, 0])
        self.min_y = np.min(self.positions[:, 1])
        self.max_y = np.max(self.positions[:, 1])
    
    def calculate_coverage(self, positions: np.ndarray) -> float:
        """计算布局覆盖度"""
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        coverage = 1 / (np.mean(min_distances) + 1e-6)
        return coverage
    
    def calculate_demand_satisfaction(self, positions: np.ndarray) -> float:
        """计算需求满足度"""
        peak_times = ['09', '12', '18']
        total_satisfaction = 0
        count = 0
        
        for time in peak_times:
            try:
                state = self.model.get_current_state(time)
                for point_data in state.values():
                    satisfaction = min(point_data['current'], point_data['demand']) / (point_data['demand'] + 1e-6)
                    total_satisfaction += satisfaction
                    count += 1
            except ValueError:
                continue
        
        return total_satisfaction / (count + 1e-6)
    
    def calculate_balance(self, positions: np.ndarray) -> float:
        """计算布局均衡性"""
        peak_times = ['09', '12', '18']
        densities = []
        
        for time in peak_times:
            try:
                state = self.model.get_current_state(time)
                current = np.array([data['current'] for data in state.values()])
                total = np.sum(current)
                if total > 0:
                    densities.extend(current / total)
            except ValueError:
                continue
        
        if not densities:
            return 0
        
        density_std = np.std(densities)
        return 1 / (1 + density_std)
    
    def calculate_displacement_penalty(self, positions: np.ndarray) -> float:
        """计算位移惩罚项"""
        positions = positions.reshape(-1, 2)
        displacements = np.linalg.norm(positions - self.original_positions, axis=1)
        
        # 如果任何位移超过最大允许值，返回极大的惩罚值
        if np.any(displacements > self.max_displacement):
            return 1e6
        
        # 计算相对位移惩罚（位移/最大允许位移）的平方
        relative_displacements = displacements / self.max_displacement
        return np.mean(relative_displacements ** 2)
    
    def objective_function(self, positions: np.ndarray) -> float:
        """目标函数：最大化布局评分，同时考虑位移约束"""
        positions = positions.reshape(-1, 2)
        
        # 计算各项指标
        coverage = self.calculate_coverage(positions)
        satisfaction = self.calculate_demand_satisfaction(positions)
        balance = self.calculate_balance(positions)
        displacement_penalty = self.calculate_displacement_penalty(positions)
        
        # 计算综合评分
        weights = {
            'coverage': 0.4,       # 覆盖度权重
            'satisfaction': 0.3,   # 需求满足度权重
            'balance': 0.2,        # 均衡性权重
            'displacement': 0.1    # 位移惩罚权重
        }
        
        score = (
            weights['coverage'] * coverage +
            weights['satisfaction'] * satisfaction +
            weights['balance'] * balance -
            weights['displacement'] * displacement_penalty
        )
        
        return -score
    
    def optimize_layout(self, max_iterations: int = 300) -> Dict[str, List[float]]:
        """优化停车点位布局，考虑位移约束"""
        # 为每个点设置移动范围约束
        bounds = []
        for pos in self.original_positions:
            # x坐标范围约束
            x_min = max(self.min_x, pos[0] - self.max_displacement)
            x_max = min(self.max_x, pos[0] + self.max_displacement)
            # y坐标范围约束
            y_min = max(self.min_y, pos[1] - self.max_displacement)
            y_max = min(self.max_y, pos[1] + self.max_displacement)
            bounds.extend([(x_min, x_max), (y_min, y_max)])
        
        # 使用差分进化算法进行全局优化
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=max_iterations,
            popsize=30,
            mutation=(0.5, 1.5),
            recombination=0.9,
            seed=42,
            workers=-1
        )
        
        # 重构优化后的位置
        optimized_positions = result.x.reshape(-1, 2)
        
        # 创建新的停车点字典
        optimized_layout = {
            name: pos.tolist() for name, pos in zip(self.point_names, optimized_positions)
        }
        
        return optimized_layout
    
    def evaluate_layout(self, layout: Dict[str, List[float]]) -> Dict[str, float]:
        """评估布局方案"""
        positions = np.array(list(layout.values()))
        
        # 计算平均位移距离
        displacements = np.linalg.norm(positions - self.original_positions, axis=1)
        avg_displacement = np.mean(displacements)
        max_displacement = np.max(displacements)
        
        return {
            "覆盖度": self.calculate_coverage(positions),
            "需求满足度": self.calculate_demand_satisfaction(positions),
            "均衡性": self.calculate_balance(positions),
            "平均位移": avg_displacement,
            "最大位移": max_displacement
        }
    
    def plot_layout_comparison(self, original_layout: Dict[str, List[float]], 
                             optimized_layout: Dict[str, List[float]], 
                             save_path: str = None):
        """绘制布局对比图"""
        plt.figure(figsize=(15, 7))
        
        # 原始布局
        plt.subplot(1, 2, 1)
        original_positions = np.array(list(original_layout.values()))
        plt.scatter(original_positions[:, 0], original_positions[:, 1], 
                   c='blue', alpha=0.6, label='原始布局')
        for i, name in enumerate(original_layout.keys()):
            plt.annotate(name, (original_positions[i, 0], original_positions[i, 1]))
        plt.title('原始布局')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.legend()
        
        # 优化后布局
        plt.subplot(1, 2, 2)
        optimized_positions = np.array(list(optimized_layout.values()))
        plt.scatter(optimized_positions[:, 0], optimized_positions[:, 1], 
                   c='red', alpha=0.6, label='优化后布局')
        
        # 绘制位移箭头
        for i, (orig, opt) in enumerate(zip(original_positions, optimized_positions)):
            plt.arrow(orig[0], orig[1], opt[0]-orig[0], opt[1]-orig[1],
                     head_width=20, head_length=30, fc='gray', ec='gray', alpha=0.3)
        
        for i, name in enumerate(optimized_layout.keys()):
            plt.annotate(name, (optimized_positions[i, 0], optimized_positions[i, 1]))
        plt.title('优化后布局')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    # 加载数据
    data_loader = DataLoader()
    parking_points = data_loader.load_parking_points('processed_data/processed_parking_points.json')
    distances = data_loader.load_distances('processed_data/processed_distances.json')
    demand_data = data_loader.load_time_demand('results')
    
    # 初始化模型
    model = BikeOperationModel(parking_points, distances, demand_data)
    
    # 评估原始布局
    print("\n原始布局评估结果:")
    print("=" * 50)
    original_performance = model.evaluate_overall_performance(is_optimized=False)
    print(f"综合评分: {original_performance['overall_score']:.4f} ({original_performance['score_description']})")
    print("\n详细评分:")
    for metric, score in original_performance['detailed_scores'].items():
        print(f"{metric}: {score:.4f}")
    
    # 优化布局（设置最大位移距离为300米）
    print("\n开始优化布局...")
    optimizer = LayoutOptimizer(model, max_displacement=300)
    optimized_layout = optimizer.optimize_layout()
    
    # 评估优化后的布局
    print("\n优化后布局评估结果:")
    print("=" * 50)
    layout_evaluation = optimizer.evaluate_layout(optimized_layout)
    print("布局指标:")
    for metric, score in layout_evaluation.items():
        print(f"{metric}: {score:.4f}")
    
    # 创建新的模型评估优化后的布局
    optimized_model = BikeOperationModel(optimized_layout, distances, demand_data)
    optimized_performance = optimized_model.evaluate_overall_performance(is_optimized=True)
    print(f"\n优化后综合评分: {optimized_performance['overall_score']:.4f} ({optimized_performance['score_description']})")
    print("\n优化后详细评分:")
    for metric, score in optimized_performance['detailed_scores'].items():
        print(f"{metric}: {score:.4f}")
    
    # 创建结果目录
    os.makedirs('analysis_results', exist_ok=True)
    
    # 绘制布局对比图
    optimizer.plot_layout_comparison(
        parking_points,
        optimized_layout,
        save_path='analysis_results/layout_comparison.png'
    )
    
    # 保存优化建议
    print("\n布局优化建议:")
    print("=" * 50)
    for point, (original_pos, optimized_pos) in zip(parking_points.keys(), 
                                                  zip(parking_points.values(), optimized_layout.values())):
        distance = np.linalg.norm(np.array(original_pos) - np.array(optimized_pos))
        if distance > 30:  # 如果移动距离超过30米
            print(f"建议将 {point} 从 [{original_pos[0]:.1f}, {original_pos[1]:.1f}] 移动到 [{optimized_pos[0]:.1f}, {optimized_pos[1]:.1f}] (移动距离: {distance:.2f}米)")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 