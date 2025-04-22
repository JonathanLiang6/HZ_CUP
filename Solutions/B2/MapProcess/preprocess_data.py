import json
import pandas as pd
import os
from typing import Dict, Tuple

def create_location_mapping() -> Dict[str, str]:
    """创建位置名称映射"""
    return {
        "二食堂": "5",
        "菊苑1栋": "P9",
        "梅苑1栋": "P3",
        "工程中心": "P12",
        "南门": "P1",
        "东门": "P11",
        "教学4楼": "P6",
        "北门": "P15",
        "教学2楼": "P7",
        "网球场": "P10",
        "校医院": "P4",
        "三食堂": "P2",
        "一食堂": "P8",
        "计算机学院": "P13",
        "体育馆": "P14"
    }

def preprocess_parking_points(parking_points_file: str, 
                            output_file: str,
                            location_mapping: Dict[str, str]):
    """预处理停车点数据"""
    with open(parking_points_file, 'r', encoding='utf-8') as f:
        points_data = json.load(f)
    
    # 创建反向映射
    reverse_mapping = {v: k for k, v in location_mapping.items()}
    
    # 转换数据格式
    processed_data = {}
    for point_id, coords in points_data.items():
        if point_id in reverse_mapping:
            name = reverse_mapping[point_id]
            processed_data[name] = coords
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return processed_data

def preprocess_distances(distances_file: str,
                        output_file: str,
                        location_mapping: Dict[str, str]):
    """预处理距离数据"""
    with open(distances_file, 'r', encoding='utf-8') as f:
        distances_data = json.load(f)
    
    # 创建反向映射
    reverse_mapping = {v: k for k, v in location_mapping.items()}
    
    # 转换数据格式
    processed_data = {}
    for key, distance in distances_data.items():
        loc1, loc2 = key.split(',')
        if loc1 in reverse_mapping and loc2 in reverse_mapping:
            name1 = reverse_mapping[loc1]
            name2 = reverse_mapping[loc2]
            processed_data[f"{name1},{name2}"] = distance
            processed_data[f"{name2},{name1}"] = distance  # 确保双向距离
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return processed_data

def preprocess_demand(demand_file: str,
                     output_file: str):
    """预处理需求数据"""
    df = pd.read_csv(demand_file)
    
    # 计算需求
    df['需求'] = df['理想配置'] - df['现有数量']
    
    # 保存处理后的数据
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    return df

def main():
    # 创建输出目录
    os.makedirs('processed_data', exist_ok=True)
    
    # 创建位置映射
    location_mapping = create_location_mapping()
    
    # 预处理数据
    print("预处理停车点数据...")
    parking_points = preprocess_parking_points(
        'map_data/parking_points.json',
        'processed_data/processed_parking_points.json',
        location_mapping
    )
    
    print("预处理距离数据...")
    distances = preprocess_distances(
        'map_data/distances.json',
        'processed_data/processed_distances.json',
        location_mapping
    )
    
    print("预处理需求数据...")
    demand_df = preprocess_demand(
        'results/bike_demand_07_00_20250419_142113.csv',
        'processed_data/processed_demand.csv'
    )
    
    print("\n数据预处理完成！")
    print(f"处理后的停车点数量: {len(parking_points)}")
    print(f"处理后的距离数据数量: {len(distances)}")
    print(f"处理后的需求数据数量: {len(demand_df)}")
    
    # 打印一些示例数据
    print("\n示例数据:")
    print("\n停车点示例:")
    for name, coords in list(parking_points.items())[:3]:
        print(f"{name}: {coords}")
    
    print("\n距离示例:")
    for key, distance in list(distances.items())[:3]:
        print(f"{key}: {distance}")
    
    print("\n需求示例:")
    print(demand_df.head())

if __name__ == "__main__":
    main() 