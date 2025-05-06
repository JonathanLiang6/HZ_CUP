# 共享单车运营数据分析与优化系统

## 一、项目概述

本项目聚焦于共享单车运营领域，旨在构建一套全面且高效的数据分析与优化系统。通过对停车点位置、距离、需求等多维度数据的深入分析，实现对共享单车运营效率的精准评估，并提供针对性的布局优化和维护调度方案，以提升共享单车的整体运营质量和服务水平。

## 二、项目结构

项目主要包含四个核心模块，每个模块负责不同的功能：

### B1 - 数据预处理与分析

- **ImprovedBikeDataAnalyzer**：该类是数据预处理与分析的核心，负责从 Excel 文件中加载共享单车分布统计数据，并进行一系列预处理操作，包括数据清洗、格式转换、特殊值处理（如 “200+”）、时间格式标准化以及无效数据删除等。同时，还具备参数优化功能，通过优化 “200+” 的估计值和流动性因子，提高数据处理的准确性。最后，生成预测表格和数据分析总结报告，为后续分析提供基础数据。

### B2 - 需求预测与路线优化

- **RequestModel**：负责加载需求数据，支持从文件或默认数据加载。同时，对模型性能进行评估，计算 MAE、MSE、RMSE 等指标，并将预测结果、理想值和评估指标保存到文件中。
- **MapProcess**：包含地图数据的加载、保存和预处理功能。能够计算需求、合并相近和共线的路线段，优化路线规划，提高运输效率。
- **Routes**：加载预处理后的数据，创建位置对象，验证数据有效性，并提供路线优化功能，为共享单车的调度提供最优方案。

### B3 - 运营效率评估与布局优化

- **BikeOperationModel**：是运营效率评估的核心类，通过初始化停车点位置、距离和需求数据，构建距离矩阵和存储需求数据。可以获取指定时间的运营状态，评估指定时间的运营效率，包括可用率、需求满足率、使用率、布局均衡性等指标，并计算综合评分。同时，还能评估总体运营表现，分析高峰期数据特征。
- **LayoutOptimizer**：负责优化停车点位布局，考虑位移约束，通过计算布局覆盖度、需求满足度、布局均衡性和位移惩罚项，构建目标函数，使用优化算法找到最优布局方案，并评估布局方案的优劣，提供布局优化建议。

### B4 - 维护调度优化

- **MaintenanceOptimizer**：使用蚁群优化算法对故障车辆的回收路线进行优化，通过局部搜索进一步改进解的质量。同时，更新自适应参数，提高算法的搜索效率。最终输出优化结果，包括总回收车辆数、总行程数、预计总时间等信息，并将详细结果保存到 JSON 文件中，同时绘制路线图。

## 三、代码文件说明

### B1 - bike_analysis.py

- **load_and_preprocess_data**：从 Excel 文件中加载数据，进行数据清洗、格式转换、特殊值处理和时间格式标准化等操作，删除无效数据。
- **optimize_parameters**：运行参数优化，使用`BikeCountOptimizer`类找到 “200+” 的最优估计值和最优流动性因子。
- **process_with_optimal_params**：使用优化后的参数处理数据，更新数据处理逻辑和流动性因子。
- **generate_summary_report**：生成数据分析总结报告，计算每个点位的统计信息，并保存到 Excel 文件中。

### B2 - RequestModel/request.py

- **load_data**：加载需求数据，支持从文件或默认数据加载。
- **evaluate_model**：评估模型性能，计算 MAE、MSE、RMSE 等指标。
- **save_results**：将预测结果、理想值和评估指标保存到文件中。

### B2 - MapProcess/preprocess_data.py

- **preprocess_demand**：预处理需求数据，计算需求并保存处理后的数据。

### B2 - MapProcess/map_processor.py

- **optimize_routes**：优化路线，合并相近和共线的路线段。
- **load_data**：加载已保存的停车点信息、路线信息和距离矩阵。
- **save_data**：保存处理后的数据，包括停车点信息、路线信息和距离矩阵。

### B2 - Routes/bicycle_scheduling_improved.py

- **load_data**：加载预处理后的数据，创建位置对象，验证数据有效性。
- **optimize_routes**：优化路线，为共享单车的调度提供最优方案。

### B3 - bike_operation_model.py

- **BikeOperationModel**：初始化共享单车运营模型，获取指定时间的运营状态，评估运营效率和总体运营表现。
- **LayoutOptimizer**：优化停车点位布局，考虑位移约束，评估布局方案。

### B4 - maintenance_optimization.py

- **main**：加载数据，创建优化器，运行蚁群优化算法，输出优化结果，保存详细结果到 JSON 文件中，并绘制路线图。
- **local_search**：对蚁群算法得到的解进行局部搜索优化。
- **evaluate_solution**：评估解的质量，综合考虑故障率差距、回收率和时间利用率。
- **update_adaptive_params**：根据局部搜索的结果更新自适应参数。

## 四、使用方法

### 1. 环境准备

确保你已经安装了以下 Python 库：

```plaintext
numpy
pandas
json
scipy
matplotlib
seaborn
```

### 2. 数据准备

将相关数据文件（如 Excel 文件、JSON 文件、CSV 文件等）放置在正确的目录下，确保文件路径与代码中的配置一致。

### 3. 运行代码

- **B1 模块**：运行bike_analysis.py中的`main`函数，进行数据预处理、参数优化、数据处理、可视化、插值、总量估算、预测表格生成和数据分析总结等操作。

```python
from HZ_CUP2025B.Solutions.B1.bike_analysis import main
main()
```

- **B2 模块**：根据具体需求调用`RequestModel`、`MapProcess`和`Routes`中的相关函数，进行需求预测、路线优化等操作。

```python
# 示例：加载数据并评估模型
from HZ_CUP2025B.Solutions.B2.RequestModel.request import RequestModel
model = RequestModel()
df = model.load_data()
predictions = model.predict()
metrics = model.evaluate_model(df, predictions)
model.save_results(predictions, ideal_values, metrics)
```

- **B3 模块**：运行bike_operation_model.py中的`main`函数，进行运营效率评估和布局优化。

```python
from HZ_CUP2025B.Solutions.B3.bike_operation_model import main
main()
```

- **B4 模块**：运行maintenance_optimization.py中的`main`函数，进行维护调度优化。


```python
from HZ_CUP2025B.Solutions.B4.maintenance_optimization import main
main()
```
## 五、注意事项

- 确保数据文件的格式和内容符合代码的要求，特别是列名和数据类型。
- 在运行优化算法时，可能需要根据实际情况调整参数，以获得更好的优化效果。
- 部分代码中使用了中文标签，确保你的环境支持中文显示，避免出现乱码问题。

## 六、贡献与反馈

如果你对本项目有任何建议或发现了问题，请随时提交 issue 或 pull request，我们将竭诚为你服务。