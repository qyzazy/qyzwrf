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

# 定义要绘制热图的指标
metrics = ['Test Accuracy', 'RMSE', 'R^2', 'MAE', 'RRMSE', 'EVS']
colors = ['magma', 'plasma', 'viridis',  'cividis', 'coolwarm', 'inferno']  # 每个热图不同的颜色

# 创建热图
for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 8))
    heatmap_data = data[[metric]].unstack()  # 转换数据格式为适合热图

    # 调整列的顺序
    if 'Beijing' in heatmap_data.columns:
        # 获取当前的列索引
        column_order = heatmap_data.columns.tolist()
        # 移动Beijing到Hainan之后和Hubei之前
        column_order.remove('Beijing')
        new_column_order = []
        for col in column_order:
            new_column_order.append(col)
            if col == 'Hubei':
                new_column_order.append('Beijing')  # 在Hubei之后添加Beijing
        # 确保Hainan在Beijing之前
        if 'Hainan' in new_column_order:
            hainan_index = new_column_order.index('Hainan')
            new_column_order.insert(hainan_index + 1, 'Beijing')  # 在Hainan之后添加Beijing
        heatmap_data = heatmap_data[new_column_order]

    sns.heatmap(heatmap_data, annot=True, cmap=colors[i], fmt=".2f", cbar=True)

    # 修改x轴文本标签
    plt.xticks(rotation=45, ha='right')  # 将x轴文本旋转45度并调整对齐方式
    plt.yticks(rotation=0)  # y轴文本保持水平

    # 在右上角添加文本标注
    plt.title(f'{metric}', fontsize=16, fontweight='bold')  # 标题加粗
    plt.xlabel('Central Area', fontsize=12)
    plt.ylabel('Center coordinates', fontsize=12)

    # 获取去掉前缀的标签
    central_area_labels = [label[1] for label in heatmap_data.columns]  # 提取元组中的第二个元素

    # 调整x轴短竖指示位置
    plt.xticks(ticks=[x + 0.5 for x in range(len(central_area_labels))], labels=central_area_labels, rotation=45, ha='right')  # 向右移动半格

    # 右上角文本标注加粗和斜体
    plt.text(0.95, 0.95, f'Mean: {data[metric].mean():.2f}\nMax: {data[metric].max():.2f}\nMin: {data[metric].min():.2f}',
             ha='right', va='top', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', fontstyle='italic',
             bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'E:\\zuotu\\heatmap_{metric}.png')  # 保存热图为PNG文件
    plt.show()
