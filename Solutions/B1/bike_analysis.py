"""
共享单车数据分析系统

本系统用于分析校园共享单车的分布情况，主要功能包括：
1. 数据加载和预处理
2. 数据插值和预测
3. 可视化分析
4. 参数优化
5. 报告生成

使用的主要方法：
- PCHIP插值：保持数据单调性的插值方法
- 粒子群优化(PSO)：用于优化参数
- 核密度估计：用于估算200+的实际值
- 移动平均：用于数据平滑
- 统计方法：用于估算总量和流动性因子
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import warnings
from datetime import datetime, time
import matplotlib.font_manager as fm
import pyswarms as ps  # 需要安装：pip install pyswarms


class BikeDataAnalyzer:
    """共享单车数据分析器基类"""
    
    def __init__(self, file_path):
        """
        初始化分析器
        
        参数:
            file_path (str): Excel数据文件路径
        """
        self.file_path = file_path
        self.df = None  # 处理后的数据框
        self.results = {}  # 存储插值结果
        self.validation_errors = []  # 存储验证错误
        self.mobility_factor = 0.3  # 默认流动性因子

    def load_and_preprocess_data(self):
        """
        加载并预处理数据
        
        处理步骤：
        1. 读取Excel文件
        2. 数据清洗和格式转换
        3. 处理特殊值（如200+）
        4. 时间格式标准化
        5. 删除无效数据
        """
        try:
            # 读取Excel数据
            print(f"正在读取文件: {self.file_path}")
            self.raw_df = pd.read_excel(self.file_path)
            print(f"原始数据形状: {self.raw_df.shape}")
            print("原始数据列名:", self.raw_df.columns.tolist())

            # 检查数据格式
            if 'Unnamed: 0' not in self.raw_df.columns or 'Unnamed: 1' not in self.raw_df.columns:
                print("警告：未找到预期的列名")
                print("实际列名:", self.raw_df.columns.tolist())
                return

            # 数据预处理
            self.df = self.raw_df.copy()
            
            # 删除第一列（周几）并重命名时间列
            self.df = self.df.drop('Unnamed: 0', axis=1)
            self.df = self.df.rename(columns={'Unnamed: 1': '时间'})

            # 将宽格式转换为长格式（便于分析）
            self.df = pd.melt(self.df,
                            id_vars=['时间'],
                            var_name='点位',
                            value_name='数量')

            # 处理200+的情况
            self.df['原始数量'] = self.df['数量'].copy()
            self.df['数量'] = self.df['数量'].replace('200+', '250')
            self.df['数量'] = pd.to_numeric(self.df['数量'], errors='coerce')

            # 时间格式标准化
            def convert_time(t):
                """将时间转换为小时数"""
                if isinstance(t, time):
                    return t.hour + t.minute / 60
                elif isinstance(t, str):
                    try:
                        time_obj = datetime.strptime(t, '%H:%M:%S').time()
                        return time_obj.hour + time_obj.minute / 60
                    except ValueError:
                        print(f"警告：无法解析时间字符串: {t}")
                        return np.nan
                else:
                    print(f"警告：未知的时间格式: {type(t)} - {t}")
                    return np.nan

            self.df['时间'] = self.df['时间'].apply(convert_time)

            # 删除无效数据
            self.df = self.df.dropna(subset=['时间', '数量'])

            # 输出数据统计信息
            print("\n数据预处理完成:")
            print(f"总记录数: {len(self.df)}")
            print(f"有效点位数: {self.df['点位'].nunique()}")
            if len(self.df) > 0:
                print(f"时间范围: {self.df['时间'].min():.2f} - {self.df['时间'].max():.2f}")
                print("\n各点位的数据量:")
                print(self.df.groupby('点位').size())

        except Exception as e:
            print(f"数据预处理错误: {str(e)}")
            import traceback
            print("详细错误信息:")
            print(traceback.format_exc())
            raise

    def interpolate_data(self):
        """
        使用PCHIP插值方法处理数据
        
        PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) 插值：
        - 保持数据单调性
        - 避免龙格现象
        - 适合处理可能存在噪声的数据
        """
        plt.figure(figsize=(15, 8))
        interpolated_count = 0

        for point in self.df['点位'].unique():
            data = self.df[self.df['点位'] == point].sort_values('时间')
            data = data.dropna(subset=['数量'])

            if len(data) >= 3:
                try:
                    # 确保时间点严格递增
                    data = data.drop_duplicates(subset=['时间'], keep='first')
                    x = data['时间'].values
                    y = data['数量'].values

                    if len(x) >= 3 and np.all(np.diff(x) > 0):
                        # 使用PCHIP插值
                        pchip = PchipInterpolator(x, y)

                        # 存储结果
                        self.results[point] = {
                            'interpolator': pchip,
                            'x_range': (min(x), max(x)),
                            'data_points': len(x),
                            'original_x': x,
                            'original_y': y
                        }

                        # 可视化
                        t_test = np.linspace(min(x), max(x), 100)
                        plt.plot(t_test, pchip(t_test), label=f"{point}({len(x)}点)")
                        plt.scatter(x, y, alpha=0.5)
                        interpolated_count += 1

                except Exception as e:
                    self.validation_errors.append(f"{point}插值失败: {str(e)}")

        if interpolated_count == 0:
            print("警告：没有足够的数据点进行插值")
            return

        print(f"成功对{interpolated_count}个点位进行了插值")

        # 设置图表样式
        plt.title('共享单车时段数量分布（带原始数据点）')
        plt.xlabel('时间（小时）')
        plt.ylabel('单车数量')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('bike_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_prediction_table(self):
        """生成预测结果表格，确保总数恒定"""
        if not self.results:
            print("错误：没有可用的插值结果")
            return pd.DataFrame()

        # 定义固定的时间点和点位顺序
        time_targets = [7.0, 9.0, 12.0, 14.0, 18.0, 21.0, 23.0]
        locations = [
            '东门', '南门', '北门', '一食堂', '二食堂', '三食堂',
            '梅苑1栋', '菊苑1栋', '教学2楼', '教学4楼', '计算机学院',
            '工程中心', '网球场', '体育馆', '校医院'
        ]

        # 创建DataFrame
        columns = ['点位'] + [f'{int(t):02d}:00' for t in time_targets]
        df = pd.DataFrame(columns=columns)
        df['点位'] = locations

        # 估算总数
        TOTAL_BIKES = self.estimate_total_bikes()

        # 填充预测值
        for point in locations:
            if point in self.results:
                try:
                    interpolator = self.results[point]['interpolator']
                    x_range = self.results[point]['x_range']

                    for t in time_targets:
                        col_name = f'{int(t):02d}:00'

                        # 使用插值函数
                        if x_range[0] <= t <= x_range[1]:
                            val = float(max(0, interpolator(t)))
                        else:
                            val = float(
                                max(0, interpolator(x_range[0]) if t < x_range[0] else interpolator(x_range[1])))

                        df.loc[df['点位'] == point, col_name] = int(round(val))

                except Exception as e:
                    self.validation_errors.append(f"{point}预测失败: {str(e)}")

        # 调整各点位的值以确保总数恒定
        for col in df.columns:
            if col != '点位':
                current_total = df[col].sum()
                if current_total != 0:  # 避免除以零
                    # 按比例调整每个点位的值
                    adjustment_factor = TOTAL_BIKES / current_total
                    df[col] = df[col].apply(lambda x: int(round(x * adjustment_factor)))

                    # 处理舍入误差
                    final_total = df[col].sum()
                    if final_total != TOTAL_BIKES:
                        # 将差值添加到最大值的位置
                        diff = TOTAL_BIKES - final_total
                        max_idx = df[col].idxmax()
                        df.loc[max_idx, col] = df.loc[max_idx, col] + diff

        # 添加总数行
        total_row = {'点位': '总数'}
        for col in df.columns:
            if col != '点位':
                total_row[col] = int(df[col].sum())
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        # 保存结果
        if len(df) > 0:
            with pd.ExcelWriter('表1.xlsx', engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='表1')

                # 获取工作表
                worksheet = writer.sheets['表1']

                # 设置格式
                from openpyxl.styles import Border, Side, Alignment, Font
                thin_border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )

                # 应用格式
                center_aligned = Alignment(horizontal='center', vertical='center')
                for row in worksheet.iter_rows(min_row=1, max_row=len(df) + 1,
                                               min_col=1, max_col=len(columns)):
                    for cell in row:
                        cell.border = thin_border
                        cell.alignment = center_aligned
                        if cell.row == len(df) + 1:  # 总数行
                            cell.font = Font(bold=True)

                # 设置列宽和行高
                for idx, col in enumerate(df.columns):
                    column_letter = chr(65 + idx)
                    worksheet.column_dimensions[column_letter].width = 12
                for idx in range(1, len(df) + 2):
                    worksheet.row_dimensions[idx].height = 25

            print(f"已生成预测结果表格，包含{len(df)}个点位的数据")
            print("\n预测结果预览:")
            print(df)
        else:
            print("警告：没有生成预测结果，数据不足")

        return df

    def estimate_total_bikes(self):
        """
        使用统计方法估算总投放量
        
        方法：
        1. 计算每个时间点的总数
        2. 使用均值±2倍标准差筛选有效数据
        3. 计算筛选后数据的均值作为估计值
        """
        if not self.df.empty:
            # 按时间点分组计算总和
            time_totals = self.df.groupby('时间')['数量'].sum()

            # 计算均值和标准差
            mean_total = time_totals.mean()
            std_total = time_totals.std()

            # 使用均值±2倍标准差的范围内的数据重新计算均值
            valid_totals = time_totals[
                (time_totals >= mean_total - 2 * std_total) &
                (time_totals <= mean_total + 2 * std_total)
            ]

            estimated_total = int(round(valid_totals.mean()))
            print(f"\n估算结果:")
            print(f"预估校园共享单车总数：{estimated_total}辆")
            print(f"标准差：{std_total:.2f}")
            return estimated_total

    def visualize_data(self):
        """
        生成数据可视化图表
        
        生成三种图表：
        1. 各点位数据量分布图
        2. 总体时间趋势图（带标准差范围）
        3. 各点位箱线图
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 各点位数据量分布图
        plt.figure(figsize=(12, 6))
        data_counts = self.df.groupby('点位').size()
        data_counts.plot(kind='bar')
        plt.title('各点位数据量分布')
        plt.xlabel('点位')
        plt.ylabel('数据点数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 总体时间趋势图
        plt.figure(figsize=(12, 6))
        time_trend = self.df.groupby('时间')['数量'].agg(['mean', 'std']).reset_index()
        plt.plot(time_trend['时间'], time_trend['mean'], 'b-', linewidth=2, label='平均值')
        plt.fill_between(
            time_trend['时间'],
            time_trend['mean'] - time_trend['std'],
            time_trend['mean'] + time_trend['std'],
            alpha=0.3,
            label='标准差范围'
        )
        plt.title('全校区单车数量时间趋势')
        plt.xlabel('时间（小时）')
        plt.ylabel('平均单车数量')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('time_trend.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 箱线图
        plt.figure(figsize=(15, 8))
        self.df.boxplot(column='数量', by='点位', figsize=(15, 8))
        plt.title('各点位单车数量分布')
        plt.xlabel('点位')
        plt.ylabel('单车数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('location_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self):
        """生成数据分析总结报告"""
        # 计算每个点位的统计信息
        summary_stats = self.df.groupby('点位')['数量'].agg([
            ('平均数量', 'mean'),
            ('最大数量', 'max'),
            ('最小数量', 'min'),
            ('标准差', 'std'),
            ('数据点数', 'count')
        ]).round(2)

        # 重置索引，使点位成为一列
        summary_stats = summary_stats.reset_index()

        # 保存到Excel，移除encoding参数
        summary_stats.to_excel('数据分析总结.xlsx', index=False)
        return summary_stats

    def clean_and_interpolate_data(self):
        """数据清洗和线性插值"""
        try:
            # 1. 首先处理200+的情况，使用临近时间点的均值进行估计
            self.df['数量'] = self.df['数量'].replace('200+', np.nan)
            self.df['数量'] = pd.to_numeric(self.df['数量'], errors='coerce')

            # 2. 按点位分组进行线性插值
            for point in self.df['点位'].unique():
                point_data = self.df[self.df['点位'] == point].copy()
                point_data = point_data.sort_values('时间')

                # 使用线性插值填充缺失值
                point_data['数量'] = point_data['数量'].interpolate(method='linear')

                # 更新原始数据框
                self.df.loc[self.df['点位'] == point, '数量'] = point_data['数量']

        except Exception as e:
            print(f"数据清洗和插值错误: {str(e)}")
            raise

    def interpolate_specific_times(self):
        """对特定时间点进行插值"""
        target_times = [7.0, 9.0, 12.0, 14.0, 18.0, 21.0, 23.0]
        results = pd.DataFrame()
        results['点位'] = self.df['点位'].unique()

        for point in results['点位']:
            point_data = self.df[self.df['点位'] == point].sort_values('时间')

            # 使用三次样条插值，保持数据平滑性
            if len(point_data) >= 4:  # 至少需要4个点才能进行三次样条插值
                spline = CubicSpline(point_data['时间'], point_data['数量'])

                # 对每个目标时间点进行插值
                for t in target_times:
                    col_name = f'{int(t):02d}:00'
                    # 确保插值结果非负
                    results.loc[results['点位'] == point, col_name] = max(0, spline(t))
            else:
                # 如果点太少，使用线性插值
                for t in target_times:
                    col_name = f'{int(t):02d}:00'
                    # 找到最近的两个时间点
                    prev_time = point_data[point_data['时间'] <= t]['时间'].max()
                    next_time = point_data[point_data['时间'] >= t]['时间'].min()

                    if pd.notna(prev_time) and pd.notna(next_time):
                        prev_val = point_data[point_data['时间'] == prev_time]['数量'].iloc[0]
                        next_val = point_data[point_data['时间'] == next_time]['数量'].iloc[0]
                        # 线性插值
                        weight = (t - prev_time) / (next_time - prev_time)
                        val = prev_val + weight * (next_val - prev_val)
                        results.loc[results['点位'] == point, col_name] = max(0, val)

    def smooth_data(self):
        """使用移动平均对数据进行平滑处理"""
        window_size = 3  # 可以根据需要调整窗口大小

        for point in self.df['点位'].unique():
            point_data = self.df[self.df['点位'] == point].sort_values('时间')

            # 使用移动平均进行平滑
            smoothed_values = point_data['数量'].rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean()

            # 更新原始数据
            self.df.loc[self.df['点位'] == point, '数量'] = smoothed_values

    def validate_and_adjust_results(self, results_df):
        """验证和调整插值结果"""
        # 确保每个时间点的总数接近估算的总量
        estimated_total = self.estimate_total_bikes()

        for col in results_df.columns:
            if col != '点位':
                current_total = results_df[col].sum()
                if current_total > 0:
                    # 按比例调整
                    adjustment_factor = estimated_total / current_total
                    results_df[col] = results_df[col].apply(
                        lambda x: max(0, round(x * adjustment_factor))
                    )

        # 添加验证信息
        print("\n验证结果:")
        for col in results_df.columns:
            if col != '点位':
                print(f"{col} 总数: {results_df[col].sum()}")

        return results_df

    def estimate_overflow_value(self):
        """
        使用统计方法估算200+的实际值
        
        方法：
        1. 获取接近200的数值
        2. 使用核密度估计获取分布
        3. 结合密度最高点和置信区间上限
        """
        try:
            # 获取接近200的所有有效数值
            near_200_values = self.df[
                (self.df['原始数量'] != '200+') &
                (pd.to_numeric(self.df['原始数量'], errors='coerce') >= 150)
            ]['原始数量'].astype(float)

            if len(near_200_values) > 0:
                # 使用核密度估计获取分布
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(near_200_values)

                # 在200以上的区间采样点
                sample_points = np.linspace(200, 300, 1000)
                densities = kde(sample_points)

                # 找到密度最高的点作为估计值
                overflow_estimate = sample_points[np.argmax(densities)]

                # 使用75%置信区间上限作为保守估计
                confidence_value = np.percentile(near_200_values, 75)

                # 取两者的加权平均
                final_estimate = 0.7 * overflow_estimate + 0.3 * confidence_value

                return round(final_estimate, 2)
            return 250.0  # 默认值
        except Exception as e:
            print(f"估算200+值时出错: {str(e)}")
            return 250.0

    def estimate_mobility_factor(self):
        """
        使用统计方法估算流动性因子
        
        方法：
        1. 计算相邻时间点的变化率
        2. 使用中位数和四分位数进行加权平均
        3. 确保因子在合理范围内
        """
        try:
            # 计算每个时间点相邻点位之间的车辆数变化
            mobility_factors = []
            time_points = sorted(self.df['时间'].unique())

            for t1, t2 in zip(time_points[:-1], time_points[1:]):
                data_t1 = self.df[self.df['时间'] == t1]['数量']
                data_t2 = self.df[self.df['时间'] == t2]['数量']

                if len(data_t1) > 0 and len(data_t2) > 0:
                    total_t1 = data_t1.sum()
                    total_t2 = data_t2.sum()
                    if total_t1 > 0:
                        change_rate = abs(total_t2 - total_t1) / total_t1
                        mobility_factors.append(change_rate)

            if mobility_factors:
                # 使用中位数作为基准
                median_mobility = np.median(mobility_factors)
                # 使用四分位距来评估波动范围
                q1 = np.percentile(mobility_factors, 25)
                q3 = np.percentile(mobility_factors, 75)

                # 使用加权平均得到最终估计
                weighted_mobility = (0.5 * median_mobility +
                                   0.3 * q1 +
                                   0.2 * q3)

                # 确保因子在合理范围内
                return max(0.2, min(0.4, round(weighted_mobility, 2)))

            return 0.3  # 默认值
        except Exception as e:
            print(f"估算流动性因子时出错: {str(e)}")
            return 0.3


class BikeCountOptimizer:
    """共享单车数量优化器"""
    
    def __init__(self, data_analyzer):
        """
        初始化优化器
        
        参数:
            data_analyzer (BikeDataAnalyzer): 数据分析器实例
        """
        self.analyzer = data_analyzer
        self.bounds = {
            'overflow_estimate': (200, 300),
            'mobility_factor': (0.2, 0.4)
        }
        # 预计算并缓存常用值
        self.unique_points = self.analyzer.df['点位'].unique()
        self.time_points = np.linspace(7, 23, 25)

    def _compute_error(self, overflow_value, mobility_factor):
        """
        计算优化目标函数值
        
        参数:
            overflow_value (float): 200+的估计值
            mobility_factor (float): 流动性因子
            
        返回:
            float: 误差值
        """
        try:
            # 使用向量化操作替代循环
            temp_df = self.analyzer.df.copy()
            temp_df['数量'] = pd.to_numeric(
                temp_df['原始数量'].replace('200+', str(overflow_value)),
                errors='coerce'
            )

            total_error = 0
            point_data_cache = {}  # 缓存点位数据

            # 只计算关键时间点
            for t in self.time_points[::2]:
                point_counts = []

                for point in self.unique_points:
                    if point not in point_data_cache:
                        point_data = temp_df[temp_df['点位'] == point]
                        if len(point_data) >= 3:
                            point_data_cache[point] = {
                                'x': point_data['时间'].values,
                                'y': point_data['数量'].values
                            }

                    if point in point_data_cache:
                        data = point_data_cache[point]
                        try:
                            if len(data['x']) >= 3 and len(np.unique(data['x'])) >= 3:
                                pchip = PchipInterpolator(data['x'], data['y'])
                                estimated_count = pchip(t)
                                if not np.isnan(estimated_count):
                                    point_counts.append(max(0, estimated_count))
                        except:
                            continue

                if point_counts:
                    adjusted_total = sum(point_counts) * (1 - mobility_factor)

                    # 简化惩罚项计算
                    if adjusted_total < 0 or adjusted_total > 5000:
                        total_error += 1000

                    # 只在必要时计算标准差
                    if len(point_counts) > 5:
                        std_dev = np.std(point_counts)
                        if std_dev > 100:
                            total_error += std_dev

            return float(total_error)

        except Exception as e:
            return float('inf')

    def optimize(self):
        """
        使用粒子群优化算法优化参数
        
        返回:
            dict: 包含最优参数的字典
        """
        try:
            options = {
                'c1': 0.5,  # 个体学习因子
                'c2': 0.3,  # 社会学习因子
                'w': 0.9    # 惯性权重
            }

            bounds = (
                [self.bounds['overflow_estimate'][0], self.bounds['mobility_factor'][0]],
                [self.bounds['overflow_estimate'][1], self.bounds['mobility_factor'][1]]
            )

            # 创建PSO优化器
            optimizer = ps.single.GlobalBestPSO(
                n_particles=10,
                dimensions=2,
                options=options,
                bounds=bounds
            )

            # 执行优化
            best_cost, best_pos = optimizer.optimize(
                self._compute_error,
                iters=50,
                verbose=False
            )

            return {
                'optimal_overflow_value': float(best_pos[0]),
                'optimal_mobility_factor': float(best_pos[1]),
                'fitness_score': float(best_cost)
            }

        except Exception as e:
            return {
                'optimal_overflow_value': 250.0,
                'optimal_mobility_factor': 0.3,
                'fitness_score': float('inf')
            }


class ImprovedBikeDataAnalyzer(BikeDataAnalyzer):
    """改进的共享单车数据分析器"""
    
    def __init__(self, file_path):
        super().__init__(file_path)
        self.optimal_params = None

    def optimize_parameters(self):
        """运行参数优化"""
        optimizer = BikeCountOptimizer(self)
        self.optimal_params = optimizer.optimize()
        print(f"优化结果：")
        print(f"200+最优估计值: {self.optimal_params['optimal_overflow_value']:.2f}")
        print(f"最优流动性因子: {self.optimal_params['optimal_mobility_factor']:.2f}")

    def process_with_optimal_params(self):
        """使用优化后的参数处理数据"""
        if self.optimal_params is None:
            self.optimize_parameters()

        # 更新数据处理逻辑
        self.df['数量'] = self.df['原始数量'].replace(
            '200+',
            str(self.optimal_params['optimal_overflow_value'])
        )
        self.df['数量'] = pd.to_numeric(self.df['数量'])

        # 使用优化后的流动性因子
        self.mobility_factor = self.optimal_params['optimal_mobility_factor']


def main():
    """主函数"""
    analyzer = ImprovedBikeDataAnalyzer('附件1-共享单车分布统计表.xlsx')

    try:
        print("开始数据处理...")
        analyzer.load_and_preprocess_data()

        if analyzer.df is None or len(analyzer.df) == 0:
            print("错误：没有有效数据，程序终止")
            return

        print("\n优化参数...")
        analyzer.optimize_parameters()

        print("\n使用优化后的参数处理数据...")
        analyzer.process_with_optimal_params()

        print("\n生成可视化图表...")
        analyzer.visualize_data()

        print("\n进行数据插值...")
        analyzer.interpolate_data()

        print("\n估算总量...")
        estimated_total = analyzer.estimate_total_bikes()

        print("\n生成预测表格...")
        prediction_table = analyzer.generate_prediction_table()

        print("\n生成数据分析总结...")
        summary = analyzer.generate_summary_report()
        print("\n数据分析总结：")
        print(summary)

        if analyzer.validation_errors:
            print("\n处理过程中的警告：")
            for error in analyzer.validation_errors:
                print(f"- {error}")

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()