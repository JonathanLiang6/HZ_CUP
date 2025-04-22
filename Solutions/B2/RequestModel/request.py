import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
def set_chinese_font():
    """设置matplotlib中文字体支持"""
    if os.name == 'nt':  # Windows系统
        # 使用微软雅黑字体
        font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=10)
        plt.rcParams['font.family'] = ['Microsoft YaHei']
        return font
    else:  # Linux/Mac系统
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

class BikeDemandPredictor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.time_cols = ["07:00", "09:00", "12:00", "14:00", "18:00", "21:00", "23:00"]
        self.font = set_chinese_font()  # 初始化时设置中文字体
        # 设置默认的图表风格
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def load_data(self):
        """加载数据，支持从文件或默认数据加载"""
        if self.data_path and os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
        else:
            data = {
                "点位": ["东门", "南门", "北门", "一食堂", "二食堂", "三食堂", "梅苑1栋", "菊苑1栋",
                        "教学2楼", "教学4楼", "计算机学院", "工程中心", "网球场", "体育馆", "校医院"],
                "07:00": [59, 89, 45, 6, 238, 13, 183, 195, 36, 49, 6, 93, 23, 6, 21],
                "09:00": [114, 105, 45, 79, 63, 69, 154, 136, 47, 50, 12, 75, 54, 50, 9],
                "12:00": [115, 70, 137, 76, 78, 88, 70, 88, 67, 173, 16, 51, 5, 19, 9],
                "14:00": [74, 42, 94, 7, 125, 17, 28, 90, 87, 197, 116, 101, 30, 6, 48],
                "18:00": [57, 166, 113, 63, 87, 91, 39, 153, 52, 9, 60, 46, 76, 41, 9],
                "21:00": [122, 130, 66, 79, 76, 68, 172, 170, 45, 45, 10, 62, 3, 13, 1],
                "23:00": [12, 70, 26, 114, 164, 165, 152, 169, 40, 30, 23, 56, 21, 5, 15]
            }
            df = pd.DataFrame(data)
        return df

    def remove_outliers(self, df):
        """移除异常值"""
        for col in self.time_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        return df

    def predict_demand(self, df, weights=None, window_size=3):
        """需求预测模型，支持自定义权重和窗口大小"""
        if weights is None:
            weights = np.array([0.5, 0.3, 0.2])  # 默认权重
        
        predictions = {}
        for loc in df["点位"]:
            loc_data = df[df["点位"] == loc][self.time_cols].values[0]
            predicted = []
            
            for i in range(len(loc_data)):
                if i < window_size - 1:
                    predicted.append(int(np.mean(loc_data[:i + 1])))
                else:
                    window = loc_data[max(0, i - window_size + 1):i + 1]
                    predicted.append(int(np.dot(window, weights[:len(window)])))
            
            predictions[loc] = dict(zip(self.time_cols, predicted))
        return predictions

    def calculate_ideal_values(self, predictions, min_safety_margin=0.2):
        """计算理想配置值，支持自定义最小安全边际"""
        ideal = {}
        for loc, preds in predictions.items():
            values = list(preds.values())
            volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else min_safety_margin
            ideal[loc] = {t: int(p * (1 + max(min_safety_margin, volatility))) for t, p in preds.items()}
        return ideal

    def evaluate_model(self, df, predictions):
        """评估模型性能"""
        mae = 0
        mse = 0
        count = 0
        
        for loc in df["点位"]:
            actual = df[df["点位"] == loc][self.time_cols].values[0]
            predicted = list(predictions[loc].values())
            
            mae += np.sum(np.abs(np.array(actual) - np.array(predicted)))
            mse += np.sum((np.array(actual) - np.array(predicted)) ** 2)
            count += len(actual)
        
        mae /= count
        mse /= count
        rmse = np.sqrt(mse)
        
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        }

    def visualize_predictions(self, df, predictions, ideal_values, save_path=None):
        """可视化预测结果"""
        plt.figure(figsize=(15, 10))
        
        for i, loc in enumerate(df["点位"]):
            plt.subplot(5, 3, i + 1)
            actual = df[df["点位"] == loc][self.time_cols].values[0]
            predicted = list(predictions[loc].values())
            ideal = list(ideal_values[loc].values())
            
            plt.plot(self.time_cols, actual, 'b-', label='实际值')
            plt.plot(self.time_cols, predicted, 'r--', label='预测值')
            plt.plot(self.time_cols, ideal, 'g:', label='理想值')
            plt.title(loc, fontproperties=self.font)
            plt.xticks(rotation=45, fontproperties=self.font)
            plt.legend(prop=self.font)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, predictions, ideal_values, metrics, output_dir='results'):
        """保存结果到文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存预测结果
        with open(f'{output_dir}/predictions_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
            
        # 保存理想值
        with open(f'{output_dir}/ideal_values_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(ideal_values, f, ensure_ascii=False, indent=2)
            
        # 保存评估指标
        with open(f'{output_dir}/metrics_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def format_results(self, df, predictions, ideal_values):
        """格式化结果，按时间分类显示每个点位的现有数量和需求数量"""
        formatted_results = {}
        
        for time in self.time_cols:
            time_results = []
            for loc in df["点位"]:
                current = df[df["点位"] == loc][time].values[0]
                predicted = predictions[loc][time]
                ideal = ideal_values[loc][time]
                
                time_results.append({
                    "点位": loc,
                    "现有数量": current,
                    "预测需求": predicted,
                    "理想配置": ideal
                })
            
            # 按现有数量降序排序
            time_results.sort(key=lambda x: x["现有数量"], reverse=True)
            formatted_results[time] = time_results
        
        return formatted_results

    def save_formatted_results(self, formatted_results, output_dir='results'):
        """保存格式化后的结果到CSV文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for time, results in formatted_results.items():
            # 将时间中的冒号替换为下划线，避免文件名问题
            time_str = time.replace(':', '_')
            df = pd.DataFrame(results)
            df.to_csv(f'{output_dir}/bike_demand_{time_str}_{timestamp}.csv', 
                     index=False, encoding='utf-8-sig')

    def plot_heatmap(self, df, save_path=None):
        """绘制自行车需求热力图"""
        plt.figure(figsize=(12, 8))
        data = df[self.time_cols].values.astype(float)  # 确保数据类型为float
        locations = df['点位'].values
        
        sns.heatmap(data, xticklabels=self.time_cols, yticklabels=locations,
                   cmap='YlOrRd', annot=True, fmt='.0f',  # 使用.0f格式显示整数
                   cbar_kws={'label': '自行车数量'})
        
        plt.title('各点位不同时段自行车需求热力图', fontproperties=self.font)
        plt.xlabel('时间', fontproperties=self.font)
        plt.ylabel('地点', fontproperties=self.font)
        
        # 设置刻度标签字体
        plt.xticks(fontproperties=self.font)
        plt.yticks(fontproperties=self.font)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_time_series_analysis(self, df, save_path=None):
        """绘制时序分析图"""
        plt.figure(figsize=(15, 10))
        
        # 计算每个时间点的统计数据
        mean_values = df[self.time_cols].mean()
        std_values = df[self.time_cols].std()
        
        plt.plot(self.time_cols, mean_values, 'b-', label='平均需求')
        plt.fill_between(self.time_cols, 
                        mean_values - std_values,
                        mean_values + std_values,
                        alpha=0.2, color='blue', label='标准差范围')
        
        plt.title('全天自行车需求时序分析', fontproperties=self.font)
        plt.xlabel('时间', fontproperties=self.font)
        plt.ylabel('自行车数量', fontproperties=self.font)
        plt.xticks(rotation=45, fontproperties=self.font)
        plt.legend(prop=self.font)
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_time_series.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_distribution_analysis(self, df, save_path=None):
        """绘制需求分布分析图"""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # 箱线图
        ax1 = fig.add_subplot(gs[0, 0])
        df[self.time_cols].boxplot(ax=ax1)
        ax1.set_title('各时段需求分布箱线图', fontproperties=self.font)
        ax1.set_xticklabels(self.time_cols, rotation=45, fontproperties=self.font)
        ax1.set_ylabel('自行车数量', fontproperties=self.font)
        
        # 小提琴图
        ax2 = fig.add_subplot(gs[0, 1])
        data_melted = pd.melt(df[self.time_cols])
        sns.violinplot(x='variable', y='value', data=data_melted, ax=ax2)
        ax2.set_title('各时段需求分布小提琴图', fontproperties=self.font)
        ax2.set_xticklabels(self.time_cols, rotation=45, fontproperties=self.font)
        ax2.set_ylabel('自行车数量', fontproperties=self.font)
        
        # 总体分布直方图
        ax3 = fig.add_subplot(gs[1, :])
        all_values = df[self.time_cols].values.flatten()
        sns.histplot(all_values, kde=True, ax=ax3)
        ax3.set_title('总体需求分布直方图', fontproperties=self.font)
        ax3.set_xlabel('自行车数量', fontproperties=self.font)
        ax3.set_ylabel('频次', fontproperties=self.font)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_location_comparison(self, df, save_path=None):
        """绘制各点位需求对比图"""
        plt.figure(figsize=(15, 8))
        
        # 计算每个点位的平均需求
        location_means = df[self.time_cols].mean(axis=1)
        location_std = df[self.time_cols].std(axis=1)
        
        # 排序后绘制条形图
        sorted_indices = location_means.argsort()
        locations = df['点位'].values[sorted_indices]
        means = location_means.values[sorted_indices]
        stds = location_std.values[sorted_indices]
        
        bars = plt.bar(locations, means, yerr=stds, capsize=5)
        plt.title('各点位平均需求对比', fontproperties=self.font)
        plt.xlabel('地点', fontproperties=self.font)
        plt.ylabel('平均自行车数量', fontproperties=self.font)
        plt.xticks(rotation=45, ha='right', fontproperties=self.font)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontproperties=self.font)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_location_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_all(self, df, predictions, ideal_values, save_path=None):
        """生成所有可视化图表"""
        if save_path is None:
            save_path = 'bike_demand_analysis.png'
            
        # 原有的预测可视化
        self.visualize_predictions(df, predictions, ideal_values, save_path)
        
        # 新增的可视化
        self.plot_heatmap(df, save_path)
        self.plot_time_series_analysis(df, save_path)
        self.plot_distribution_analysis(df, save_path)
        self.plot_location_comparison(df, save_path)

def main():
    # 创建预测器实例
    predictor = BikeDemandPredictor()
    
    # 加载数据
    df = predictor.load_data()
    
    # 移除异常值
    df = predictor.remove_outliers(df)
    
    # 需求预测
    predictions = predictor.predict_demand(df)
    
    # 计算理想配置值
    ideal_values = predictor.calculate_ideal_values(predictions)
    
    # 评估模型
    metrics = predictor.evaluate_model(df, predictions)
    
    # 格式化结果
    formatted_results = predictor.format_results(df, predictions, ideal_values)
    
    # 保存格式化结果
    predictor.save_formatted_results(formatted_results)
    
    # 打印结果
    print("\n模型评估指标：")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print("\n各时段单车需求分析：")
    for time, results in formatted_results.items():
        print(f"\n{time}时段：")
        print("点位\t\t现有数量\t预测需求\t理想配置")
        print("-" * 50)
        for result in results:
            print(f"{result['点位']}\t{result['现有数量']}\t\t{result['预测需求']}\t\t{result['理想配置']}")
    
    # 生成所有可视化图表
    predictor.visualize_all(df, predictions, ideal_values, 'bike_demand_analysis.png')
    
    # 保存原始结果
    predictor.save_results(predictions, ideal_values, metrics)

if __name__ == "__main__":
    main()