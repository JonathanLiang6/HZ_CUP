import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
import pickle
import os
from matplotlib.widgets import Button
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class MapProcessor:
    def __init__(self, image_path: str):
        """初始化地图处理器"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 转换BGR到RGB
        self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        
        # 存储标记的点和路线
        self.parking_points: Dict[str, Tuple[int, int]] = {}  # 停车点坐标
        self.routes: List[List[Tuple[int, int]]] = []  # 提取的路线
        self.scale = 2000.0 / self.height  # 比例尺（图片高度对应2000米）
        
        # 用于交互式标记
        self.current_point_name = None
        self.fig = None
        self.ax = None
        self.point_counter = 1

    def extract_yellow_routes(self):
        """提取黄色路线（使用局部区域检测）"""
        try:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            
            # 定义黄色的HSV范围（调整以更好地匹配图片中的黄色）
            lower_yellow = np.array([20, 50, 150])
            upper_yellow = np.array([35, 255, 255])
            
            # 创建黄色掩码
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 使用自适应阈值进行局部处理
            window_size = 51  # 必须是奇数
            yellow_mask = cv2.adaptiveThreshold(
                yellow_mask,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                window_size,
                2
            )
            
            # 形态学操作改善路线连续性
            kernel = np.ones((3,3), np.uint8)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            
            # 使用霍夫变换检测直线段
            lines = cv2.HoughLinesP(
                yellow_mask,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10
            )
            
            # 将检测到的线段转换为路线
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    self.routes.append([(x1, y1), (x2, y2)])
                
                # 优化路线合并
                self.optimize_routes()
            
            print(f"已提取 {len(self.routes)} 条路线")
            
        except Exception as e:
            print(f"提取路线时出错: {e}")
            self.routes = []

    def optimize_routes(self, distance_threshold=20, angle_threshold=30):
        """优化路线（合并相近和共线的路线段）"""
        if not self.routes:
            return
        
        def get_angle(line1, line2):
            """计算两条线段的夹角"""
            dx1 = line1[1][0] - line1[0][0]
            dy1 = line1[1][1] - line1[0][1]
            dx2 = line2[1][0] - line2[0][0]
            dy2 = line2[1][1] - line2[0][1]
            angle1 = np.arctan2(dy1, dx1)
            angle2 = np.arctan2(dy2, dx2)
            angle = np.abs(angle1 - angle2) * 180 / np.pi
            return min(angle, 180 - angle)
        
        # 使用numpy数组进行快速计算
        routes_array = np.array(self.routes)
        optimized_routes = []
        used = set()
        
        for i in range(len(self.routes)):
            if i in used:
                continue
                
            current_route = list(self.routes[i])
            used.add(i)
            
            # 查找可以合并的路线
            for j in range(i + 1, len(self.routes)):
                if j in used:
                    continue
                    
                route2 = self.routes[j]
                
                # 检查端点距离和角度
                for p1 in [current_route[0], current_route[-1]]:
                    for p2 in [route2[0], route2[-1]]:
                        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                        if dist < distance_threshold:
                            angle = get_angle(current_route, route2)
                            if angle < angle_threshold:
                                # 合并路线
                                current_route.extend(route2)
                                used.add(j)
                                break
            
            optimized_routes.append(current_route)
        
        self.routes = optimized_routes

    def on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # 左键点击
            x, y = int(event.xdata), int(event.ydata)
            point_name = f"P{self.point_counter}"
            self.point_counter += 1
            
            # 添加点和标签
            self.parking_points[point_name] = (x, y)
            self.ax.plot(x, y, 'ro', markersize=10)
            self.ax.annotate(f"{point_name}", (x, y), 
                           xytext=(5, 5), textcoords='offset points')
            self.fig.canvas.draw()
            
            print(f"已添加停车点: {point_name} 在坐标 ({x}, {y})")
        
        elif event.button == 3:  # 右键点击删除最近的点
            if not self.parking_points:
                return
            
            x, y = event.xdata, event.ydata
            closest_point = None
            min_dist = float('inf')
            
            for name, (px, py) in self.parking_points.items():
                dist = np.sqrt((px-x)**2 + (py-y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = name
            
            if min_dist < 20:  # 删除20像素范围内的最近点
                del self.parking_points[closest_point]
                self.refresh_plot()
                print(f"已删除停车点: {closest_point}")

    def refresh_plot(self):
        """刷新图片显示"""
        self.ax.clear()
        self.ax.imshow(self.image)
        
        # 显示路线
        for route in self.routes:
            route_array = np.array(route)
            self.ax.plot(route_array[:, 0], route_array[:, 1], 'y-', linewidth=2)
        
        # 显示停车点
        for name, (x, y) in self.parking_points.items():
            self.ax.plot(x, y, 'ro', markersize=10)
            self.ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points')
        
        self.ax.set_title('点击添加停车点 (左键添加, 右键删除)')
        self.fig.canvas.draw()

    def interactive_marking(self):
        """交互式标记界面"""
        self.fig, self.ax = plt.subplots(figsize=(15, 15))
        self.ax.imshow(self.image)
        self.ax.set_title('点击添加停车点 (左键添加, 右键删除)')
        
        # 显示已提取的路线
        for route in self.routes:
            route_array = np.array(route)
            self.ax.plot(route_array[:, 0], route_array[:, 1], 'y-', linewidth=2)
        
        # 添加完成按钮
        plt.subplots_adjust(bottom=0.2)
        ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
        button = Button(ax_button, '完成标记')
        button.on_clicked(lambda event: plt.close(self.fig))
        
        # 连接鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def calculate_distances(self) -> Dict[Tuple[str, str], float]:
        """计算停车点之间的实际距离（米）"""
        distances = {}
        for name1 in self.parking_points:
            for name2 in self.parking_points:
                if name1 != name2:
                    p1 = self.parking_points[name1]
                    p2 = self.parking_points[name2]
                    # 计算像素距离
                    pixel_distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    # 转换为实际距离（米）
                    real_distance = pixel_distance * self.scale
                    distances[(name1, name2)] = real_distance
        return distances

    def save_data(self, output_dir: str = 'map_data'):
        """保存处理后的数据"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 保存停车点信息
        with open(f'{output_dir}/parking_points.json', 'w', encoding='utf-8') as f:
            # 转换坐标为列表以便JSON序列化
            points_data = {name: list(coord) for name, coord in self.parking_points.items()}
            json.dump(points_data, f, ensure_ascii=False, indent=2)
            
        # 保存路线信息
        with open(f'{output_dir}/routes.pkl', 'wb') as f:
            pickle.dump(self.routes, f)
            
        # 保存距离矩阵
        distances = self.calculate_distances()
        with open(f'{output_dir}/distances.json', 'w', encoding='utf-8') as f:
            # 转换键为字符串以便JSON序列化
            distances_data = {f"{k[0]},{k[1]}": v for k, v in distances.items()}
            json.dump(distances_data, f, ensure_ascii=False, indent=2)

    def load_data(self, input_dir: str = 'map_data'):
        """加载已保存的数据"""
        try:
            # 加载停车点信息
            with open(f'{input_dir}/parking_points.json', 'r', encoding='utf-8') as f:
                points_data = json.load(f)
                self.parking_points = {name: tuple(coord) for name, coord in points_data.items()}
            
            # 加载路线信息
            with open(f'{input_dir}/routes.pkl', 'rb') as f:
                self.routes = pickle.load(f)
                
            return True
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def get_scheduler_data(self) -> Dict:
        """生成调度器可用的数据格式"""
        distances = self.calculate_distances()
        return {
            'locations': self.parking_points,
            'distances': distances,
            'scale': self.scale
        }

def main():
    try:
        # 创建地图处理器实例
        processor = MapProcessor('campus_map.jpg')
        
        # 检查是否存在已保存的数据
        if not processor.load_data():
            print("未找到已保存的数据，开始处理图片...")
            # 提取黄色路线
            processor.extract_yellow_routes()
            
            # 交互式标记停车点
            processor.interactive_marking()
            
            # 保存处理后的数据
            processor.save_data()
        
        # 获取调度器可用的数据
        scheduler_data = processor.get_scheduler_data()
        
        # 打印处理结果
        print("\n处理完成!")
        print(f"已标记的停车点数量: {len(processor.parking_points)}")
        print(f"提取的路线数量: {len(processor.routes)}")
        
        # 显示最终结果
        plt.figure(figsize=(15, 15))
        plt.imshow(processor.image)
        
        # 显示路线
        for route in processor.routes:
            route_array = np.array(route)
            plt.plot(route_array[:, 0], route_array[:, 1], 'y-', linewidth=2)
        
        # 显示停车点
        for name, (x, y) in processor.parking_points.items():
            plt.plot(x, y, 'ro', markersize=10)
            plt.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title('校园地图处理结果')
        plt.savefig('processed_map.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 