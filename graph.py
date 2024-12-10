import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取CSV文件，指定编码格式
file_path = r'E:\zuotu\111.csv'
data = pd.read_csv(file_path, encoding='gbk')  # 根据需要选择合适的编码

# 设置中心坐标和区域作为行列索引
data.set_index(['Center coordinates', 'Central Area'], inplace=True)

# 定义要绘制的指标
metrics = ['R^2', 'MAE', 'RRMSE', 'EVS']

# 创建不同的颜色
colors = sns.color_palette("Set2", len(metrics))

# 创建箱线图
plt.figure(figsize=(12, 8))
boxplot = sns.boxplot(data=data[metrics], palette=colors, width=0.6)

# 添加标题和标签
plt.xlabel('Metrics', fontsize=18)
plt.ylabel('Values', fontsize=18)

# 设置坐标轴刻度字体大小
plt.xticks(fontsize=17)  # 横坐标字体
plt.yticks(fontsize=17)  # 纵坐标字体

# 添加网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 设置y轴范围
plt.ylim(0, data[metrics].max().max() * 1.1)

# 优化图例，表示中位数、上下四分位数和异常值的符号
legend_elements = [
    plt.Line2D([0], [0], color='black', lw=4, label='Median'),
    plt.Line2D([0], [0], color='gray', marker='|', lw=0, label='Q1', markersize=10),
    plt.Line2D([0], [0], color='gray', marker='|', lw=0, label='Q3', markersize=10),
    plt.Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor='black', lw=0, label='Outliers', markersize=8)  # 异常值标识
]

# 修改图例字体大小
plt.legend(handles=legend_elements, loc='upper left', fontsize=15, title='Statistics', title_fontsize='17', bbox_to_anchor=(1, 1))

# 获取所有线条
lines = boxplot.lines

# 加粗最上面和最下面的线
for i in range(0, len(lines), 6):  # 6是每个箱体的线条数量
    lines[i].set_linewidth(2)      # 上边界
    lines[i + 1].set_linewidth(2)  # 下边界

# 保存箱线图为PNG文件
plt.tight_layout()
plt.savefig(r'E:\\zuotu\\boxplot_selected_metrics.png')  # 保存箱线图为PNG文件
plt.show()
