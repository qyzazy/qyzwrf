import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from fuzzywuzzy import fuzz

# 读取节点信息（.csv）
nodes_df = pd.read_csv(r'E:\VS\data\node.csv')
# 确保 ID 列为字符串类型
nodes_df['ID'] = nodes_df['ID'].astype(str)
# 读取关系信息（.csv）
edges_df = pd.read_csv(r'E:\VS\data\edge.csv')
# 打印 DataFrame 的前几行以确认读取成功
print(nodes_df.head())
print(edges_df.head())

# 对 `label` 和 `category` 进行独热编码
encoder = OneHotEncoder(sparse_output=False)
label_encoded = encoder.fit_transform(nodes_df[['label']])
category_encoded = encoder.fit_transform(nodes_df[['category']])

# 对 `name` 列进行文本处理
vectorizer = CountVectorizer()
name_vector = vectorizer.fit_transform(nodes_df['name']).toarray()

# 将不同的特征结合成一个 DataFrame
features = pd.DataFrame(name_vector)
features = pd.concat([features, pd.DataFrame(label_encoded), pd.DataFrame(category_encoded)], axis=1)
# 加入 ID 列
features['ID'] = nodes_df['ID'].values
print(features.head())

# 创建有向图
G = nx.from_pandas_edgelist(edges_df, 'start_id', 'end_id', create_using=nx.DiGraph())


# GAT 模型定义
class GAT(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=num_heads, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 准备数据
x = torch.tensor(features.drop(columns='ID').values, dtype=torch.float)
# 将 start_id 和 end_id 转换为相应的行索引
start_indices = edges_df['start_id'].map(lambda x: features[features['ID'] == x].index[0])
end_indices = edges_df['end_id'].map(lambda x: features[features['ID'] == x].index[0])
# 创建 edge_index
edge_index = torch.tensor(list(zip(start_indices.tolist(), end_indices.tolist())), dtype=torch.long).t().contiguous()
# 创建PyG数据对象
data = Data(x=x, edge_index=edge_index)

# 标签与掩码
labels = nodes_df['label'].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
# 转换为torch张量
labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

# 划分训练和测试集
num_nodes = len(labels_tensor)
indices = np.arange(num_nodes)
np.random.shuffle(indices)
train_size = int(0.8 * num_nodes)
train_indices = indices[:train_size]
test_indices = indices[train_size:]
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = 1
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_indices] = 1
train_labels = labels_tensor[train_mask]
test_labels = labels_tensor[test_mask]

# 训练和测试的设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(label_encoder.classes_)

# 初始化模型
model = GAT(num_features=x.shape[1], hidden_dim=16, num_classes=num_classes, num_heads=8).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)


# 训练过程
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], train_labels)
    loss.backward()
    optimizer.step()
    return loss.item()


# 测试过程
def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[test_mask] == test_labels).sum()
        acc = int(correct) / int(test_mask.sum())
    return acc, pred[test_mask].cpu().numpy()


# 训练循环
for epoch in range(900):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 循环进行预测
while True:
    # 测试精度
    accuracy, test_predictions = test()
    print(f'Test Accuracy: {accuracy:.4f}')

    # 生成新案例
    new_center_coordinates = input("请输入中心坐标（例如：116.5 °E, 38.5 °N）：")
    new_simulation_area = input("请输入模拟区域名称（例如：Beijing）：")

    # 输入参数化方案
    print("请输入参数化方案，若无输入则输入#，输入完成后在后面输入 'run'，点击Enter结束输入：")
    print("可选参数化方案包括：")
    parameterization_options = [
        "Microphysical processes：",
        "Cumulus convection parameterization：",
        "Long radiation：",
        "Short radiation：",
        "Land surface processes：",
        "Boundary layer：",
        "Near ground layer："
    ]
    input_schemes = []
    for option in parameterization_options:
        value = input(option)
        if value.strip() == '#':
            continue  # 跳过无输入的情况
        input_schemes.append(value.strip())

    # 结束输入
    while True:
        line = input("输入 'run' 结束输入：")
        if line.strip().lower() == 'run':
            break

    # 打印输入的参数化方案
    print("您输入的参数化方案为：", input_schemes)

    # 找到所有 `label` 列为 "Title" 的节点
    title_nodes = nodes_df[nodes_df['label'] == "Title"]

    # 找到与输入中心坐标和模拟区域的标题节点
    target_titles = []
    for index, row in title_nodes.iterrows():
        if fuzz.partial_ratio(new_center_coordinates, row['name']) > 60 or \
                fuzz.partial_ratio(new_simulation_area, row['name']) > 40:
            target_titles.append(row['ID'])

    # 基于标题节点查找相连的其他参数设置和方案
    recommended_params_and_schemes = {
        'Parameters': {},
        'Schemes': {}
    }


    # 提取网格配置
    def extract_grid_configuration(nodes_df):
        grid_configurations = {}
        for index, row in nodes_df.iterrows():
            if 'Horizontal resolution' in row['label']:
                value = row['name']
                if '(d01)' in value:
                    grid_configurations.setdefault('d01', value)
                elif '(d02)' in value:
                    grid_configurations.setdefault('d02', value)
                elif '(d03)' in value:
                    grid_configurations.setdefault('d03', value)
                elif '(d04)' in value:
                    grid_configurations.setdefault('d04', value)
            elif 'Grid points' in row['label']:
                value = row['name']
                if '(d01)' in value:
                    grid_configurations.setdefault('d01_grid', value)
                elif '(d02)' in value:
                    grid_configurations.setdefault('d02_grid', value)
                elif '(d03)' in value:
                    grid_configurations.setdefault('d03_grid', value)
                elif '(d04)' in value:
                    grid_configurations.setdefault('d04_grid', value)
        return grid_configurations


    # 提取网格配置
    grid_configurations = extract_grid_configuration(nodes_df)


    # 判断嵌套模式的层数
    def determine_nesting_layers(nesting_str):
        layers = 0
        if "Single layer nested" in nesting_str:
            layers = 1
        elif "Double nested" in nesting_str:
            layers = 2
        elif "Three layer" in nesting_str or "Triple bidirectional" in nesting_str:
            layers = 3
        elif "Four layer" in nesting_str:
            layers = 4
        return layers


    # 获取嵌套模式的描述并判断层数
    nesting_pattern = ""
    vertical_layering = ""
    model_top_pressure = ""
    for title_id in target_titles:
        if title_id in G:
            neighbors = list(G.neighbors(title_id))
            for neighbor_id in neighbors:
                neighbor_row = nodes_df[nodes_df['ID'] == neighbor_id]
                if not neighbor_row.empty:
                    if 'Nesting adopted by the pattern' in neighbor_row['label'].values:
                        nesting_pattern = neighbor_row['name'].values[0]
                    if 'Vertical layering' in neighbor_row['label'].values:
                        vertical_layering = neighbor_row['name'].values[0]
                    if 'Model top pressure' in neighbor_row['label'].values:
                        model_top_pressure = neighbor_row['name'].values[0]

    # 确定推荐的层数
    num_layers = determine_nesting_layers(nesting_pattern)

    # 将嵌套模式和其他参数添加到推荐参数配置中
    recommended_params_and_schemes['Parameters']['Nesting adopted by the pattern'] = nesting_pattern
    recommended_params_and_schemes['Parameters']['Vertical layering'] = vertical_layering
    recommended_params_and_schemes['Parameters']['Model top pressure'] = model_top_pressure

    # 生成推荐参数
    for i in range(1, num_layers + 1):
        key_res = f'd0{i}'
        key_grid = f'd0{i}_grid'
        recommended_params_and_schemes['Parameters'].setdefault(f'Horizontal resolution ({key_res})',
                                                                grid_configurations.get(key_res, 'N/A'))
        recommended_params_and_schemes['Parameters'].setdefault(f'Grid points ({key_res})',
                                                                grid_configurations.get(key_grid, 'N/A'))

    # 推荐方案配置
    for option in parameterization_options:
        label = option.strip('：')  # 去掉冒号
        if label not in input_schemes:
            # 检查原始案例库中是否有该方案
            matching_rows = nodes_df[nodes_df['label'] == label]
            if not matching_rows.empty:
                recommended_params_and_schemes['Schemes'][label] = matching_rows['name'].values[0]  # 取第一个匹配的值
            else:
                recommended_params_and_schemes['Schemes'][label] = 'N/A'  # 如果没有输入，设置为 'N/A'
        else:
            # 如果用户输入了该方案，假设用户输入的值已经在 input_schemes 中
            index = input_schemes.index(label)
            recommended_params_and_schemes['Schemes'][label] = input_schemes[index]

    # 输出新案例推荐
    print("生成的新案例推荐：")
    print(f"The center coordinates: {new_center_coordinates}")
    print(f"Simulation area: {new_simulation_area}")
    print("推荐的参数配置：")
    for label, value in recommended_params_and_schemes['Parameters'].items():
        print(f"{label}: {value}")
    print("推荐的方案配置：")
    for label, value in recommended_params_and_schemes['Schemes'].items():
        print(f"{label}: {value}")

    # 假设我们有一组真实的标签来进行评估
    true_labels = test_labels.numpy()
    rmse = np.sqrt(mean_squared_error(true_labels, test_predictions))
    r2 = r2_score(true_labels, test_predictions)
    mae = mean_absolute_error(true_labels, test_predictions)
    mean_true_labels = np.mean(true_labels)
    relative_rmse = rmse / mean_true_labels if mean_true_labels != 0 else float('inf')
    explained_variance = explained_variance_score(true_labels, test_predictions)
    print(f'RMSE: {rmse:.4f}')
    print(f'Relative RMSE: {relative_rmse:.4f}')
    print(f'R^2: {r2:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'Explained Variance Score: {explained_variance:.4f}')

    # 询问用户是否继续
    continue_prediction = input("是否要继续进行新的预测？（y/n）：")
    if continue_prediction.lower() != 'y':
        break

print("程序结束。")
