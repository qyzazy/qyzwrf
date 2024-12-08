from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from fuzzywuzzy import fuzz

# Initialize the Flask app
app = Flask(__name__)

# Load data
nodes_df = pd.read_csv('node.csv') 
edges_df = pd.read_csv('edge.csv') 
nodes_df['ID'] = nodes_df['ID'].astype(str)

# Create directed graph
G = nx.from_pandas_edgelist(edges_df, 'start_id', 'end_id', create_using=nx.DiGraph())


# GAT model definition
class GAT(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=num_heads, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, edge_index), dim=1)


# Initialize model
def initialize_model():
    encoder = OneHotEncoder(sparse_output=False)
    label_encoded = encoder.fit_transform(nodes_df[['label']])
    category_encoded = encoder.fit_transform(nodes_df[['category']])

    vectorizer = CountVectorizer()
    name_vector = vectorizer.fit_transform(nodes_df['name']).toarray()

    features = pd.DataFrame(name_vector)
    features = pd.concat([features, pd.DataFrame(label_encoded), pd.DataFrame(category_encoded)], axis=1)
    features['ID'] = nodes_df['ID'].values

    x = torch.tensor(features.drop(columns='ID').values, dtype=torch.float)
    start_indices = edges_df['start_id'].map(lambda x: features[features['ID'] == x].index[0])
    end_indices = edges_df['end_id'].map(lambda x: features[features['ID'] == x].index[0])

    edge_index = torch.tensor(list(zip(start_indices.tolist(), end_indices.tolist())),
                              dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)

    labels_encoded = LabelEncoder().fit_transform(nodes_df['label'].values)
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(num_features=x.shape[1], hidden_dim=8, num_classes=len(np.unique(labels_encoded)), num_heads=4).to(
        device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # Training loop
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], labels_tensor[train_mask])
        loss.backward()
        optimizer.step()

    return model, data, labels_tensor, train_mask, test_mask


def extract_grid_configuration(nodes_df):
    grid_configurations = {}
    for index, row in nodes_df.iterrows():
        if 'Horizontal resolution' in row['label']:
            value = row['name']
            if '(d01)' in value and 'd01' not in grid_configurations:
                grid_configurations['d01'] = value
            elif '(d02)' in value and 'd02' not in grid_configurations:
                grid_configurations['d02'] = value
            elif '(d03)' in value and 'd03' not in grid_configurations:
                grid_configurations['d03'] = value
            elif '(d04)' in value and 'd04' not in grid_configurations:
                grid_configurations['d04'] = value
        elif 'Grid points' in row['label']:
            value = row['name']
            if '(d01)' in value and 'd01_grid' not in grid_configurations:
                grid_configurations['d01_grid'] = value
            elif '(d02)' in value and 'd02_grid' not in grid_configurations:
                grid_configurations['d02_grid'] = value
            elif '(d03)' in value and 'd03_grid' not in grid_configurations:
                grid_configurations['d03_grid'] = value
            elif '(d04)' in value and 'd04_grid' not in grid_configurations:
                grid_configurations['d04_grid'] = value

    return grid_configurations


def generate_recommendations(center_coordinates, simulation_area, model, data, labels_tensor, train_mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    title_nodes = nodes_df[nodes_df['label'] == "Title"]

    target_titles = [
        row['ID'] for index, row in title_nodes.iterrows()
        if fuzz.partial_ratio(center_coordinates, row['name']) > 60 or
           fuzz.partial_ratio(simulation_area, row['name']) > 40
    ]

    recommended_params_and_schemes = {'Parameters': {}, 'Schemes': {}}

    # Collect relevant configurations
    if target_titles:
        first_title = target_titles[0]

        if first_title in G:
            neighbors = list(G.neighbors(first_title))
            for neighbor_id in neighbors:
                neighbor_row = nodes_df[nodes_df['ID'] == neighbor_id]
                if not neighbor_row.empty:
                    if 'Nesting adopted by the pattern' in neighbor_row['label'].values:
                        nesting_pattern = neighbor_row['name'].values[0].strip()
                        recommended_params_and_schemes['Parameters']['Nesting adopted by the pattern'] = nesting_pattern
                    if 'Vertical layering' in neighbor_row['label'].values:
                        vertical_layering = neighbor_row['name'].values[0].strip()
                        recommended_params_and_schemes['Parameters']['Vertical layering'] = vertical_layering
                    if 'Model top pressure' in neighbor_row['label'].values:
                        model_top_pressure = neighbor_row['name'].values[0].strip()
                        recommended_params_and_schemes['Parameters']['Model top pressure'] = model_top_pressure

    # Extract grid configuration
    grid_configurations = extract_grid_configuration(nodes_df)

    # Determine number of layers from nesting pattern
    num_layers = determine_nesting_layers(
        recommended_params_and_schemes['Parameters'].get('Nesting adopted by the pattern', ''))

    # Generate grid points and resolution recommendations based on layers
    for layer in range(1, num_layers + 1):
        key_res = f'd0{layer}'
        key_grid = f'd0{layer}_grid'
        recommended_params_and_schemes['Parameters'].setdefault(
            f'Horizontal resolution ({key_res})',
            grid_configurations.get(key_res, 'N/A'))
        recommended_params_and_schemes['Parameters'].setdefault(
            f'Grid points ({key_grid})',
            grid_configurations.get(key_grid, 'N/A'))

    # Returning recommended schemes
    scheme_labels = [
        "Microphysical processes",
        "Cumulus convection parameterization",
        "Long radiation",
        "Short radiation",
        "Land surface processes",
        "Boundary layer",
        "Near ground layer"
    ]

    for title_id in target_titles:
        if title_id in G:
            neighbors = list(G.neighbors(title_id))
            for neighbor_id in neighbors:
                neighbor_row = nodes_df[nodes_df['ID'] == neighbor_id]
                if not neighbor_row.empty:
                    label = neighbor_row['label'].values[0]
                    if label in scheme_labels:
                        recommended_params_and_schemes['Schemes'][label] = neighbor_row['name'].values[0]

    return recommended_params_and_schemes


def determine_nesting_layers(nesting_pattern):
    """Determine the number of layers based on the nesting pattern description."""
    layers = 0
    if "Single layer nested" in nesting_pattern:
        layers = 1
    elif "Double nested" in nesting_pattern:
        layers = 2
    elif "Three layer" in nesting_pattern or "Triple bidirectional" in nesting_pattern:
        layers = 3
    elif "Four layer" in nesting_pattern:
        layers = 4
    return layers


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    center_coordinates = request.form['coordinates']
    simulation_area = request.form['simulation_area']

    model, data, labels_tensor, train_mask, test_mask = initialize_model()
    recommendations = generate_recommendations(center_coordinates, simulation_area, model, data, labels_tensor,
                                               train_mask)

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

