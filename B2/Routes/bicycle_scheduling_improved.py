import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
from matplotlib.font_manager import FontProperties
import seaborn as sns
import random
from datetime import datetime, timedelta
import glob

# 设置中文字体
def set_chinese_font():
    """设置matplotlib中文字体支持"""
    if os.name == 'nt':  # Windows系统
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    else:  # Linux/Mac系统
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签

@dataclass
class TimeSlot:
    start_time: datetime
    end_time: datetime
    demand_file: str  # 需求预测文件路径
    
    def load_demand(self) -> Dict[str, int]:
        """加载该时间段的需求预测数据"""
        try:
            df = pd.read_csv(self.demand_file)
            return dict(zip(df['点位'], df['预测需求']))
        except Exception as e:
            print(f"加载需求数据出错: {e}")
            return {}

@dataclass
class Location:
    name: str
    coordinates: Tuple[float, float]
    current_count: int
    ideal_count: int
    base_demand: int = 0
    
    def calculate_demand(self, time_slot: TimeSlot) -> int:
        """计算特定时间段的需求量"""
        demand_data = time_slot.load_demand()
        if self.name in demand_data:
            return demand_data[self.name] - self.current_count
        return self.base_demand

class RouteOptimizer:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.max_route_length = 6  # 每条路线最多访问6个点
        self.vehicle_capacity = scheduler.vehicle_capacity  # 车辆容量
        self.max_time = scheduler.max_time  # 最大调度时间（小时）
        self.vehicle_speed = scheduler.vehicle_speed  # 车速（km/h）
        
    def calculate_route_time(self, route: List[str]) -> float:
        """计算路线时间（小时）"""
        if not route or len(route) < 2:
            return float('inf')
            
        total_time = 0
        for i in range(len(route) - 1):
            key = f"{route[i]},{route[i+1]}"
            distance = self.scheduler.distances.get(key, float('inf'))
            if distance == float('inf'):
                return float('inf')
            # 距离单位为米，转换为千米，除以速度(km/h)得到小时
            total_time += (distance / 1000) / self.vehicle_speed
            
        # 每个点假设需要5分钟的操作时间
        total_time += len(route) * (5/60)
        
        return total_time
        
    def evaluate_route(self, route: List[str], time_slot: TimeSlot) -> Tuple[float, int, List[Tuple[str, int]]]:
        """评估路线
        返回: (时间, 总调度量, 调度详情列表)
        调度详情列表中每项为(地点名, 调度量)，正值表示补充，负值表示取出
        """
        if not route or len(route) < 2:
            return float('inf'), 0, []
            
        time = self.calculate_route_time(route)
        if time == float('inf') or time > self.max_time:
            return float('inf'), 0, []
            
        total_moved = 0  # 总调度量
        current_load = 0  # 当前车上的车辆数
        schedule_details = []  # 调度详情
        
        # 遍历路线上的每个点
        for i in range(len(route)):
            loc_name = route[i]
            loc = self.scheduler.locations[loc_name]
            
            # 计算当前时间段的需求
            demand = loc.calculate_demand(time_slot)
            
            if demand < 0:  # 供应点
                # 从供应点取车
                available = min(-demand, self.vehicle_capacity)
                take_count = min(
                    available,
                    self.vehicle_capacity - current_load
                )
                if take_count > 0:
                    current_load += take_count
                    total_moved += take_count
                    schedule_details.append((loc_name, -take_count))
                    
            elif demand > 0:  # 需求点
                # 向需求点放车
                needed = min(demand, self.vehicle_capacity)
                put_count = min(
                    needed,
                    current_load
                )
                if put_count > 0:
                    current_load -= put_count
                    total_moved += put_count
                    schedule_details.append((loc_name, put_count))
                    
        return time, total_moved, schedule_details
        
    def find_optimal_routes(self, time_slot: TimeSlot) -> List[Tuple[List[str], float, int, List[Tuple[str, int]]]]:
        """寻找最优路线
        返回: [(路线, 时间, 调度量, 调度详情), ...]
        """
        # 获取供应点和需求点
        supply_points = [(name, -loc.calculate_demand(time_slot)) 
                        for name, loc in self.scheduler.locations.items() 
                        if loc.calculate_demand(time_slot) < 0]
        demand_points = [(name, loc.calculate_demand(time_slot)) 
                        for name, loc in self.scheduler.locations.items() 
                        if loc.calculate_demand(time_slot) > 0]
        
        # 按供应量/需求量排序
        supply_points.sort(key=lambda x: x[1], reverse=True)
        demand_points.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{time_slot.start_time.strftime('%H:%M')}-{time_slot.end_time.strftime('%H:%M')} 供应点情况:")
        for name, supply in supply_points:
            print(f"{name}: 可供应 {supply} 辆")
        
        print(f"\n{time_slot.start_time.strftime('%H:%M')}-{time_slot.end_time.strftime('%H:%M')} 需求点情况:")
        for name, demand in demand_points:
            print(f"{name}: 需求 {demand} 辆")
        
        if not supply_points or not demand_points:
            print("\n没有找到有效的供应点或需求点！")
            return []
            
        # 为每辆车生成路线
        best_routes = []
        remaining_supply = {name: supply for name, supply in supply_points}
        remaining_demand = {name: demand for name, demand in demand_points}
        
        for vehicle in range(self.scheduler.vehicle_count):
            best_route = None
            best_score = float('-inf')
            best_time = float('inf')
            best_moved = 0
            best_details = []
            
            # 尝试不同的起点
            for supply_start, supply_amount in supply_points:
                if remaining_supply.get(supply_start, 0) <= 0:
                    continue
                    
                # 构建路线
                route = [supply_start]
                current_points = set([supply_start])
                
                # 先添加需求点，再添加供应点，确保路线有效
                while len(route) < self.max_route_length:
                    if len(route) % 2 == 1:  # 添加需求点
                        candidates = [(name, demand) for name, demand in demand_points 
                                   if name not in current_points 
                                   and remaining_demand.get(name, 0) > 0]
                    else:  # 添加供应点
                        candidates = [(name, supply) for name, supply in supply_points 
                                   if name not in current_points 
                                   and remaining_supply.get(name, 0) > 0]
                        
                    if not candidates:
                        break
                        
                    # 选择最近的点
                    best_next = None
                    min_distance = float('inf')
                    current = route[-1]
                    
                    for next_point, _ in candidates:
                        key = f"{current},{next_point}"
                        distance = self.scheduler.distances.get(key, float('inf'))
                        if distance < min_distance:
                            min_distance = distance
                            best_next = next_point
                    
                    if best_next is None:
                        break
                        
                    route.append(best_next)
                    current_points.add(best_next)
                
                # 评估路线
                time, moved, details = self.evaluate_route(route, time_slot)
                if time == float('inf') or moved == 0:
                    continue
                    
                # 计算得分 = 调度量 / 时间
                score = moved / time if time > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_route = route
                    best_time = time
                    best_moved = moved
                    best_details = details
            
            if best_route:
                best_routes.append((best_route, best_time, best_moved, best_details))
                # 更新剩余供需
                for loc_name, amount in best_details:
                    if amount < 0:  # 取车
                        remaining_supply[loc_name] = remaining_supply.get(loc_name, 0) - abs(amount)
                    else:  # 放车
                        remaining_demand[loc_name] = remaining_demand.get(loc_name, 0) - amount
                        
        return best_routes

class BicycleScheduler:
    def __init__(self, 
                 locations: Dict[str, Location],
                 distances: Dict[str, float],
                 vehicle_count: int = 3,
                 vehicle_capacity: int = 20,
                 vehicle_speed: float = 25.0,
                 max_time: float = 1.0):
        self.locations = locations
        self.distances = distances
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_speed = vehicle_speed
        self.max_time = max_time
        self.graph = self._build_graph()
        self.optimizer = RouteOptimizer(self)
        
    def _build_graph(self) -> nx.Graph:
        """构建网络图"""
        G = nx.Graph()
        
        # 添加节点
        for name, loc in self.locations.items():
            G.add_node(name, 
                      pos=loc.coordinates,
                      current=loc.current_count,
                      ideal=loc.ideal_count,
                      demand=loc.base_demand)
        
        # 添加边
        for key, distance in self.distances.items():
            loc1, loc2 = key.split(',')
            G.add_edge(loc1, loc2, weight=distance)
            
        return G
    
    def optimize_routes(self, time_slot: TimeSlot) -> List[Tuple[List[str], float, int, List[Tuple[str, int]]]]:
        """优化路线"""
        return self.optimizer.find_optimal_routes(time_slot)
        
    def visualize_routes(self, routes: List[Tuple[List[str], float, int, List[Tuple[str, int]]]], 
                        time_slot: TimeSlot,
                        save_path: Optional[str] = None):
        """可视化路线"""
        set_chinese_font()
        plt.figure(figsize=(15, 15))
        
        # 获取节点位置
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # 绘制节点
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            demand = self.locations[node].calculate_demand(time_slot)
            if demand > 0:
                node_colors.append('red')  # 需要补充
            elif demand < 0:
                node_colors.append('green')  # 需要减少
            else:
                node_colors.append('blue')  # 平衡
            node_sizes.append(300 + abs(demand) * 10)
        
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.6)
        
        # 绘制边
        nx.draw_networkx_edges(self.graph, pos, 
                             edge_color='gray',
                             alpha=0.4)
        
        # 绘制路线
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for i, (route, time, moved, details) in enumerate(routes):
            if len(route) < 2:
                continue
                
            color = colors[i % len(colors)]
            # 只添加一次标签
            label_added = False
            for j in range(len(route) - 1):
                start = route[j]
                end = route[j + 1]
                if start in pos and end in pos:
                    plt.plot([pos[start][0], pos[end][0]], 
                            [pos[start][1], pos[end][1]],
                            color=color,
                            linewidth=2,
                            label=f'车辆 {i+1} (时间: {time:.2f}h, 调度: {moved}辆)' if not label_added else "")
                    label_added = True
        
        # 添加标签
        labels = {}
        for node in self.graph.nodes():
            current = self.graph.nodes[node]['current']
            ideal = self.graph.nodes[node]['ideal']
            demand = self.locations[node].calculate_demand(time_slot)
            labels[node] = f"{node}\n当前: {current}\n理想: {ideal}\n需求: {demand}"
        
        nx.draw_networkx_labels(self.graph, pos, labels,
                              font_family='SimHei',
                              font_size=8)
        
        plt.title(f'共享单车调度路线图 ({time_slot.start_time.strftime("%H:%M")}-{time_slot.end_time.strftime("%H:%M")})', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def load_data(processed_data_dir: str = 'processed_data') -> Tuple[Dict[str, Location], Dict[str, float]]:
    """加载预处理后的数据"""
    try:
        # 检查目录是否存在
        if not os.path.exists(processed_data_dir):
            raise FileNotFoundError(f"数据目录 {processed_data_dir} 不存在")
            
        # 检查必要文件是否存在
        required_files = ['processed_parking_points.json', 'processed_distances.json', 'processed_demand.csv']
        for file in required_files:
            if not os.path.exists(os.path.join(processed_data_dir, file)):
                raise FileNotFoundError(f"缺少必要文件: {file}")
        
        # 加载停车点坐标
        with open(f'{processed_data_dir}/processed_parking_points.json', 'r', encoding='utf-8') as f:
            points_data = json.load(f)
            if not points_data:
                raise ValueError("停车点坐标数据为空")
        
        # 加载距离数据
        with open(f'{processed_data_dir}/processed_distances.json', 'r', encoding='utf-8') as f:
            distances = json.load(f)
            if not distances:
                raise ValueError("距离数据为空")
        
        # 加载需求数据
        demand_df = pd.read_csv(f'{processed_data_dir}/processed_demand.csv')
        if demand_df.empty:
            raise ValueError("需求数据为空")
            
        # 验证必要列是否存在
        required_columns = ['点位', '现有数量', '理想配置', '需求']
        missing_columns = [col for col in required_columns if col not in demand_df.columns]
        if missing_columns:
            raise ValueError(f"需求数据缺少必要列: {', '.join(missing_columns)}")
        
        print("\n加载数据:")
        print(f"停车点数量: {len(points_data)}")
        print(f"距离数据数量: {len(distances)}")
        print(f"需求数据数量: {len(demand_df)}")
        
        # 创建位置对象
        locations = {}
        for _, row in demand_df.iterrows():
            name = row['点位']
            if name in points_data:
                current = int(row['现有数量'])
                ideal = int(row['理想配置'])
                base_demand = int(row['需求'])
                
                # 验证数据有效性
                if current < 0 or ideal < 0:
                    raise ValueError(f"点位 {name} 的现有数量或理想配置不能为负数")
                
                locations[name] = Location(
                    name=name,
                    coordinates=tuple(points_data[name]),
                    current_count=current,
                    ideal_count=ideal,
                    base_demand=base_demand
                )
        
        print("\n需求情况:")
        for name, loc in locations.items():
            print(f"{name}: 现有 {loc.current_count}, 理想 {loc.ideal_count}, 基础需求 {loc.base_demand}")
        
        return locations, distances
        
    except Exception as e:
        print(f"\n加载数据出错: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_time_slots(results_dir: str = 'results') -> List[TimeSlot]:
    """获取所有时间段"""
    time_slots = []
    # 获取所有需求预测文件
    demand_files = glob.glob(os.path.join(results_dir, 'bike_demand_*.csv'))
    
    # 定义时间段映射
    time_mapping = {
        '07_00': ('07:00', '09:00'),  # 早高峰
        '09_00': ('09:00', '11:00'),
        '12_00': ('12:00', '14:00'),  # 午间
        '14_00': ('14:00', '16:00'),
        '18_00': ('18:00', '20:00'),  # 晚高峰
        '21_00': ('21:00', '23:00'),
        '23_00': ('23:00', '01:00')
    }
    
    for file in demand_files:
        try:
            # 从文件名中提取时间标识
            time_key = os.path.basename(file).split('_')[2:4]
            time_key = '_'.join(time_key)
            
            if time_key in time_mapping:
                start_time_str, end_time_str = time_mapping[time_key]
                start_time = datetime.strptime(start_time_str, "%H:%M")
                end_time = datetime.strptime(end_time_str, "%H:%M")
                
                time_slots.append(TimeSlot(
                    start_time=start_time,
                    end_time=end_time,
                    demand_file=file
                ))
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            continue
    
    # 按时间排序
    time_slots.sort(key=lambda x: x.start_time)
    return time_slots

def main():
    try:
        # 加载数据
        locations, distances = load_data()
        
        # 创建调度器
        scheduler = BicycleScheduler(
            locations=locations,
            distances=distances,
            vehicle_count=3,  # 3辆调度车
            vehicle_capacity=20,  # 最大运输量20辆/次
            vehicle_speed=25.0,  # 车速25km/h
            max_time=1.0  # 最大调度时间1小时
        )
        
        # 获取所有时间段
        time_slots = get_time_slots()
        if not time_slots:
            print("未找到有效的需求预测文件！")
            return
            
        print(f"\n找到 {len(time_slots)} 个时间段的需求预测数据")
        
        # 对每个时间段进行调度
        for time_slot in time_slots:
            print(f"\n开始优化 {time_slot.start_time.strftime('%H:%M')}-{time_slot.end_time.strftime('%H:%M')} 的路线...")
            routes = scheduler.optimize_routes(time_slot)
            
            if not routes:
                print("未找到有效的调度路线！")
                continue
                
            # 评估结果
            print(f"\n{time_slot.start_time.strftime('%H:%M')}-{time_slot.end_time.strftime('%H:%M')} 调度结果:")
            total_moved = 0
            total_time = 0
            for i, (route, time, moved, details) in enumerate(routes):
                print(f"\n车辆 {i + 1}:")
                print(f"路线: {' -> '.join(route)}")
                print(f"时间: {time:.2f} 小时")
                print(f"调度量: {moved} 辆")
                print("调度详情:")
                for loc, count in details:
                    action = "补充" if count > 0 else "取出"
                    print(f"  {loc}: {action} {abs(count)} 辆")
                total_moved += moved
                total_time = max(total_time, time)
            
            print(f"\n总调度量: {total_moved} 辆")
            print(f"完成时间: {total_time:.2f} 小时")
            
            # 可视化
            scheduler.visualize_routes(routes, time_slot, 
                                     f'optimized_routes_{time_slot.start_time.strftime("%H%M")}.png')
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 