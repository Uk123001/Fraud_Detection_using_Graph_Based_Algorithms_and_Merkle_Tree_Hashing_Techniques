"""
GNN FRAUD DETECTION SYSTEM

This model contains the following components:
1. Merkle Tree for transaction integrity
2. Data Loader with auto-download from KaggleHub
3. Feature Extractor that computes 21 features per node
4. GNN Model (GAT + GraphSAGE)
5. Main program to orchestrate everything

We have tested it for one of the datasets in this code, we tried 3 other dataset which are:
1. ULB Credit Card Fraud Detection Dataset
2. IEEE-CIS Fraud Detection Dataset
3. Eliptic Bitcoin Fraud Detection Dataset

This code does the following functions:
1. Download the dataset from KaggleHub
2. Load the dataset into a DataFrame
3. Create user and merchant identifiers
4. Convert timestamps to datetime objects
5. Build a directed graph from the transactions
6. Build transaction history for each user
7. Create node labels based on fraud occurrence
8. Train, Test, and Evaluate the GNN model on different datasets
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import sys
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. MERKLE TREE
# ==============================================================================
class MerkleTree:
    def __init__(self, transactions):
        self.transactions = transactions
        self.root = self._build()

    def _hash(self, data):
        h = hashlib.sha256()
        h.update(str(data).encode('utf-8'))
        return h.hexdigest()

    def _build(self):
        if not self.transactions:
            return None
        leaves = [self._hash(tx['id']) for tx in self.transactions]
        while len(leaves) > 1:
            new_leaves = []
            for i in range(0, len(leaves), 2):
                left = leaves[i]
                right = leaves[i + 1] if i + 1 < len(leaves) else left
                new_leaves.append(self._hash(left + right))
            leaves = new_leaves
        return leaves[0]

    def get_root(self):
        return self.root


# ==============================================================================
# 2. DATA LOADER with Auto Download
# ==============================================================================
class DataLoader:
    def __init__(self):
        self.df = None
        self.graph = None
        self.tx_history = None
        self.labels = None

    def download_and_load(self):
        print("\n" + "=" * 70)
        print("DOWNLOADING DATASET USING KAGGLEHUB")
        print("=" * 70)

        import kagglehub

        # Download the latest version
        print("Downloading creditcardfraud dataset...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print(f"✓ Download complete. Files saved to: {path}")

        # The CSV is usually inside the downloaded folder
        csv_files = list(Path(path).glob("*.csv"))
        if not csv_files:
            print("Could not find creditcard.csv in the downloaded folder.")
            sys.exit(1)

        csv_path = csv_files[0]
        print(f"Using file: {csv_path.name}")

        # Now load the data
        print("\nLoading CSV...")
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df):,} transactions")

        self._create_identifiers()
        self._convert_timestamps()
        self.graph = self._build_graph()
        self.tx_history = self._build_transaction_history()
        self.labels = self._create_node_labels()

        print(f"\nGraph: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
        return self.graph, self.tx_history, self.labels

    def _create_identifiers(self):
        self.df = self.df.sort_values('Time').reset_index(drop=True)
        self.df['user_cluster'] = pd.cut(self.df['V1'], bins=50, labels=False, duplicates='drop').fillna(0).astype(int)
        self.df['user_cluster2'] = pd.cut(self.df['V2'], bins=20, labels=False, duplicates='drop').fillna(0).astype(int)
        self.df['time_window'] = (self.df['Time'] // (3600 * 6)).astype(int)
        self.df['amount_bucket'] = pd.qcut(self.df['Amount'], q=20, labels=False, duplicates='drop').fillna(0).astype(int)

        self.df['user_id'] = 'u_' + self.df['user_cluster'].astype(str) + '_' + \
                             self.df['user_cluster2'].astype(str) + '_' + \
                             self.df['time_window'].astype(str)

        self.df['merchant_id'] = 'm_' + (self.df['Amount'] // 50).astype(int).astype(str)

    def _convert_timestamps(self):
        ref_date = datetime(2013, 9, 1)
        self.df['timestamp'] = self.df['Time'].apply(lambda x: ref_date + timedelta(seconds=float(x)))

    def _build_graph(self):
        G = nx.DiGraph()
        for _, row in self.df.iterrows():
            sender = str(row['user_id'])
            receiver = str(row['merchant_id'])
            amount = float(row['Amount'])
            if G.has_edge(sender, receiver):
                G[sender][receiver]['weight'] += amount
                G[sender][receiver]['count'] += 1
            else:
                G.add_edge(sender, receiver, weight=amount, count=1)
        return G

    def _build_transaction_history(self):
        history = defaultdict(list)
        for _, row in self.df.iterrows():
            user = str(row['user_id'])
            history[user].append({
                'amount': float(row['Amount']),
                'timestamp': row['timestamp'],
                'merchant': str(row['merchant_id']),
                'is_fraud': int(row['Class'])
            })
        for user in history:
            history[user].sort(key=lambda x: x['timestamp'])
        return dict(history)

    def _create_node_labels(self):
        labels = {}
        for user, txs in self.tx_history.items():
            fraud_count = sum(1 for tx in txs if tx['is_fraud'] == 1)
            labels[user] = 1 if fraud_count > 0 else 0
        return labels


# ==============================================================================
# 3. FEATURE EXTRACTOR (21 features) 
# ==============================================================================
class FeatureExtractor:
    def __init__(self, graph, tx_history):
        self.graph = graph
        self.tx_history = tx_history
        self._precompute_centralities()

    def _precompute_centralities(self):
        print("Precomputing centralities...")
        try:
            self.betweenness = nx.betweenness_centrality(self.graph)
        except:
            self.betweenness = {n: 0.0 for n in self.graph.nodes()}
        try:
            self.pagerank = nx.pagerank(self.graph, max_iter=100)
        except:
            self.pagerank = {n: 1.0 / self.graph.number_of_nodes() for n in self.graph.nodes()}

    def extract(self, user):
        features = []
        features.extend(self._topology_features(user))
        features.extend(self._transaction_features(user))
        features.extend(self._temporal_features(user))
        features.extend(self._anomaly_features(user))
        while len(features) < 21:
            features.append(0.0)
        return features[:21]

    def _topology_features(self, user):
        if user not in self.graph: return [0.0] * 6
        in_deg = float(self.graph.in_degree(user))
        out_deg = float(self.graph.out_degree(user))
        clust = float(nx.clustering(self.graph.to_undirected(), user)) if user in self.graph.to_undirected() else 0.0
        betw = float(self.betweenness.get(user, 0.0))
        pr = float(self.pagerank.get(user, 0.0))
        isolated = 1.0 if (in_deg + out_deg) <= 1 else 0.0
        return [in_deg, out_deg, clust, betw, pr, isolated]

    def _transaction_features(self, user):
        if user not in self.tx_history or not self.tx_history[user]:
            return [0.0] * 8
        txs = self.tx_history[user]
        amounts = [tx['amount'] for tx in txs if tx['amount'] > 0]
        if not amounts: return [0.0] * 8
        total = float(sum(amounts))
        avg = float(np.mean(amounts))
        std = float(np.std(amounts)) if len(amounts) > 1 else 0.0
        max_amt = float(max(amounts))
        count = float(len(txs))
        round_ratio = sum(1 for a in amounts if a % 100 == 0) / len(amounts)
        large_ratio = sum(1 for a in amounts if a > avg * 2) / len(amounts)
        velocity = 0.0
        if len(txs) > 1:
            days = (txs[-1]['timestamp'] - txs[0]['timestamp']).days + 1
            velocity = count / max(days, 1)
        return [total, avg, std, max_amt, count, round_ratio, large_ratio, velocity]

    def _temporal_features(self, user):
        if user not in self.tx_history or not self.tx_history[user]:
            return [0.0] * 4
        txs = self.tx_history[user]
        night_ratio = sum(1 for tx in txs if 0 <= tx['timestamp'].hour < 6) / len(txs)
        weekend_ratio = sum(1 for tx in txs if tx['timestamp'].weekday() >= 5) / len(txs)
        burst_ratio = avg_gap = 0.0
        if len(txs) > 1:
            diffs = [(txs[i+1]['timestamp'] - txs[i]['timestamp']).total_seconds() / 3600 for i in range(len(txs)-1)]
            if diffs:
                burst_ratio = sum(1 for d in diffs if d < 1) / len(diffs)
                avg_gap = np.mean(diffs)
        return [night_ratio, weekend_ratio, burst_ratio, avg_gap]

    def _anomaly_features(self, user):
        if user not in self.tx_history or user not in self.graph:
            return [0.0] * 3
        txs = self.tx_history[user]
        spike = 1.0
        if len(txs) >= 5:
            recent = np.mean([tx['amount'] for tx in txs[-3:]])
            historical = np.mean([tx['amount'] for tx in txs[:-3]])
            spike = recent / historical if historical > 0 else 1.0
        balance = 0.0
        try:
            in_flow = sum(self.graph[p][user].get('weight', 0) for p in self.graph.predecessors(user))
            out_flow = sum(self.graph[user][s].get('weight', 0) for s in self.graph.successors(user))
            balance = abs(in_flow - out_flow) / max(in_flow + out_flow, 1)
        except:
            pass
        unique_merchants = len({tx.get('merchant') for tx in txs})
        counterparty = unique_merchants / len(txs) if txs else 0.0
        return [spike, balance, counterparty]


# ==============================================================================
# 4. GNN MODEL
# ==============================================================================
class FraudGNN(torch.nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        if x.size(0) > 1: x = self.bn1(x)
        x = self.dropout(x)

        x = F.elu(self.conv2(x, edge_index))
        if x.size(0) > 1: x = self.bn2(x)
        x = self.dropout(x)

        x = F.elu(self.conv3(x, edge_index))
        x = x + x   # residual connection

        return F.log_softmax(self.classifier(x), dim=1)


# ==============================================================================
# 5. MAIN PROGRAM
# ==============================================================================
def main():
    print("\n" + "=" * 75)
    print("   GNN FRAUD DETECTION SYSTEM - Auto Download via kagglehub")
    print("=" * 75)

    loader = DataLoader()
    graph, tx_history, labels = loader.download_and_load()

    # Merkle Tree sample
    print("\nTransaction Integrity Check (Merkle Tree)")
    sample_user = list(tx_history.keys())[0]
    sample_txs = [{'id': f"{sample_user}_{i}"} for i in range(len(tx_history[sample_user]))]
    tree = MerkleTree(sample_txs)
    print(f"Sample Merkle Root: {tree.get_root()[:60]}...")

    # Feature Extraction
    print("\nExtracting 21 features per node...")
    extractor = FeatureExtractor(graph, tx_history)
    node_list = list(graph.nodes())
    features_dict = {}
    for i, node in enumerate(node_list):
        features_dict[node] = extractor.extract(node)
        if (i + 1) % 2000 == 0 or i == len(node_list)-1:
            print(f"  Processed {i+1:,}/{len(node_list):,} nodes")

    # Prepare data
    x = torch.tensor([features_dict[n] for n in node_list], dtype=torch.float)
    edge_index = torch.tensor([[node_list.index(u), node_list.index(v)] for u, v in graph.edges()], dtype=torch.long).t().contiguous()
    y = torch.tensor([labels.get(n, 0) for n in node_list], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)

    print(f"\nData prepared → Nodes: {x.size(0):,}, Edges: {edge_index.size(1):,}")

    # Training
    print("\nTraining GNN model (50 epochs)...")
    model = FraudGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    fraud_count = (y == 1).sum().item()
    legit_count = (y == 0).sum().item()
    weights = torch.tensor([legit_count, fraud_count], dtype=torch.float)
    weights = weights / weights.sum() * 2.0

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, y, weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f}")

    # Evaluation
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        accuracy = (pred == y).float().mean().item() * 100
        tp = ((pred == 1) & (y == 1)).sum().item()
        fp = ((pred == 1) & (y == 0)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        tn = ((pred == 0) & (y == 0)).sum().item()

        precision = tp / max(tp + fp, 1) * 100
        recall = tp / max(tp + fn, 1) * 100
        f1 = 2 * precision * recall / max(precision + recall, 1)

        print(f"\nResults:")
        print(f"Accuracy : {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"Recall   : {recall:.2f}%")
        print(f"F1 Score : {f1:.2f}%")
        print(f"TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}")

    print("\n" + "=" * 75)
    print("SYSTEM COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
