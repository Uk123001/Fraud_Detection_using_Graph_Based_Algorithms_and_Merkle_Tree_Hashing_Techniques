"""
GNN FRAUD DETECTION SYSTEM
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DatasetDownloader:
    """Auto-downloads Credit Card Fraud dataset from Kaggle"""
    
    def __init__(self, data_dir='./fraud_datasets'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        print(f"Dataset directory: {self.data_dir.absolute()}")
    
    def download_creditcard_dataset(self):
        """Download Credit Card Fraud Detection dataset (284K transactions)"""
        dataset_name = 'mlg-ulb/creditcardfraud'
        csv_file = self.data_dir / 'creditcard.csv'
        
        if csv_file.exists():
            print(f"Dataset already exists: {csv_file}")
            return csv_file
        
        print(f"Downloading Credit Card Fraud dataset: ")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("   Downloading (144 MB, may take 2-3 minutes)")
            api.dataset_download_files(dataset_name, path=self.data_dir, unzip=True)
            
            if csv_file.exists():
                print(f"Download complete!")
                return csv_file
            else:
                raise FileNotFoundError("Download completed but file not found")
                
        except ImportError:
            print("\n ERROR: Kaggle API not installed")
            print("Install: pip install kaggle")
            print("\nSetup instructions:")
            print("1. Go to kaggle.com/account")
            print("2. Create New API Token -> downloads kaggle.json")
            print("3. Move to ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
            sys.exit(1)
            
        except Exception as e:
            print(f"\n Download error: {e}")
            print("\nManual download:")
            print(f"1. Visit: kaggle.com/datasets/{dataset_name}")
            print(f"2. Download creditcard.csv")
            print(f"3. Place in: {self.data_dir.absolute()}")
            sys.exit(1)

class RealDatasetLoader:
    """Loads and processes real fraud detection data"""
    
    def __init__(self):
        self.df = None
        self.graph = None
        self.transaction_history = None
        self.node_labels = None
    
    def load_creditcard_dataset(self, csv_path):
        """Load Credit Card dataset"""
        print("LOADING CREDIT CARD FRAUD DATASET")
        # Load CSV
        print(" Reading CSV file: ")
        try:
            self.df = pd.read_csv(csv_path)
            print(f" Loaded {len(self.df):,} transactions")
        except Exception as e:
            print(f" Error reading CSV: {e}")
            sys.exit(1)
        
        # Validate columns
        required_cols = ['Time', 'Amount', 'Class']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f" Missing columns: {missing_cols}")
            sys.exit(1)
        
        print(f" Dataset information:")
        print(f"     Total: {len(self.df):,} transactions")
        print(f"     Fraud: {self.df['Class'].sum():,} ({self.df['Class'].mean()*100:.3f}%)")
        print(f"     Legitimate: {(self.df['Class']==0).sum():,}")
        
        # Create user IDs from transaction patterns (dataset is anonymized)
        print("\n Creating user identifiers from patterns: ")
        self.df = self.df.sort_values('Time').reset_index(drop=True)
        
        # FIX 1: Use PCA features to create more realistic user clustering
        # The dataset has V1-V28 PCA features that represent transaction patterns
        pca_cols = [col for col in self.df.columns if col.startswith('V')]
        if len(pca_cols) >= 3:
            # Use first 3 PCA components for clustering
            self.df['user_cluster'] = pd.cut(self.df['V1'], bins=50, labels=False, duplicates='drop').fillna(0).astype(int)
            self.df['user_cluster2'] = pd.cut(self.df['V2'], bins=20, labels=False, duplicates='drop').fillna(0).astype(int)
        else:
            self.df['user_cluster'] = 0
            self.df['user_cluster2'] = 0
        
        # Time windows (6-hour buckets for more granularity)
        self.df['time_window'] = (self.df['Time'] // (3600 * 6)).astype(int)
        
        # Amount buckets for clustering similar transactions
        self.df['amount_bucket'] = pd.qcut(self.df['Amount'], q=20, labels=False, duplicates='drop').fillna(0).astype(int)
        
        # Create user IDs combining multiple features
        self.df['user_id'] = 'user_' + (
            self.df['user_cluster'].astype(str) + '_' + 
            self.df['user_cluster2'].astype(str) + '_' +
            self.df['time_window'].astype(str)
        )
        
        # Create merchant IDs
        self.df['merchant_id'] = 'merchant_' + (self.df['Amount'] // 50).astype(int).astype(str)
        
        print(f" Identified {self.df['user_id'].nunique():,} unique users")
        print(f" Identified {self.df['merchant_id'].nunique():,} unique merchants")
        
        # Convert timestamps
        print("\n Converting timestamps: ")
        reference_date = datetime(2013, 9, 1)
        self.df['timestamp'] = self.df['Time'].apply(
            lambda x: reference_date + timedelta(seconds=float(x))
        )
        
        # Build graph
        print("\n  Building transaction graph: ")
        self.graph = self._build_graph()
        
        # Build transaction history
        print("\n Building transaction history: ")
        self.transaction_history = self._build_tx_history()
        
        # Create labels - FIX 2: Changed threshold logic
        print("\n  Creating node labels: ")
        self.node_labels = self._create_labels()
        
        print(f"\n Processing complete!")
        print(f"     Graph nodes: {self.graph.number_of_nodes():,}")
        print(f"     Graph edges: {self.graph.number_of_edges():,}")
        print(f"     Fraud nodes: {sum(self.node_labels.values()):,}")
        
        return self.graph, self.transaction_history, self.node_labels
    
    def _build_graph(self):
        """Build directed graph from transactions"""
        G = nx.DiGraph()
        
        for idx, row in self.df.iterrows():
            sender = str(row['user_id'])
            receiver = str(row['merchant_id'])
            amount = float(row['Amount'])
            
            if G.has_edge(sender, receiver):
                G[sender][receiver]['weight'] += amount
                G[sender][receiver]['count'] += 1
            else:
                G.add_edge(sender, receiver, weight=amount, count=1)
        
        print(f"    Created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _build_tx_history(self):
        """Build per-user transaction history"""
        tx_history = defaultdict(list)
        
        for idx, row in self.df.iterrows():
            user = str(row['user_id'])
            tx_history[user].append({
                'amount': float(row['Amount']),
                'timestamp': row['timestamp'],
                'to': str(row['merchant_id']),
                'is_fraud': int(row['Class'])
            })
        
        for user in tx_history:
            tx_history[user].sort(key=lambda x: x['timestamp'])
        
        print(f"    History for {len(tx_history)} users")
        return dict(tx_history)
    
    def _create_labels(self):
        """
        FIX 2: Create node labels using ANY fraud presence
        Changed from: fraud_count > len(txs) / 2 (>50% threshold)
        Changed to: fraud_count > 0 (any fraud marks user as fraudulent)
        
        Reasoning: Original threshold was too strict - with only 0.173% fraud rate,
        almost no users would have >50% fraud transactions when grouped.
        """
        labels = {}
        
        for user, txs in self.transaction_history.items():
            fraud_count = sum(1 for tx in txs if tx['is_fraud'] == 1)
            # FIX 2: Changed threshold - any fraud activity marks the user as fraudulent
            labels[user] = 1 if fraud_count > 0 else 0
        
        fraud_users = sum(1 for label in labels.values() if label == 1)
        total_users = len(labels)
        print(f"    Labeled {total_users} nodes ({fraud_users} fraud, {total_users - fraud_users} legitimate)")
        return labels

class FeatureExtractor:
    """Extract 21 fraud detection features"""
    
    def __init__(self, graph, tx_history):
        self.graph = graph
        self.tx_history = tx_history
        print("\n Precomputing centrality metrics: ")
        self._precompute_centralities()
        print(" Centrality metrics is ready")
    
    def _precompute_centralities(self):
        """Precompute betweenness and PageRank"""
        try:
            self.betweenness = nx.betweenness_centrality(self.graph)
        except:
            self.betweenness = {n: 0.0 for n in self.graph.nodes()}
        
        try:
            self.pagerank = nx.pagerank(self.graph, max_iter=100, tol=1e-6)
        except:
            self.pagerank = {n: 1.0/self.graph.number_of_nodes() for n in self.graph.nodes()}
    
    def extract_all_features(self, user):
        """Extract all 21 features for a user"""
        features = []
        try:
            # Topology (6)
            features.extend(self._topology(user))
            # Transaction (8)
            features.extend(self._transaction(user))
            # Temporal (4)
            features.extend(self._temporal(user))
            # Anomaly (3)
            features.extend(self._anomaly(user))
            
            if len(features) != 21:
                features = (features + [0.0] * 21)[:21]
        except:
            features = [0.0] * 21
        
        return features
    
    def _topology(self, user):
        """6 topology features"""
        try:
            if user not in self.graph:
                return [0.0] * 6
            
            in_deg = float(self.graph.in_degree(user))
            out_deg = float(self.graph.out_degree(user))
            
            try:
                clust = float(nx.clustering(self.graph.to_undirected(), user))
            except:
                clust = 0.0
            
            betw = float(self.betweenness.get(user, 0.0))
            pr = float(self.pagerank.get(user, 0.0))
            isolated = 1.0 if (in_deg + out_deg) <= 1 else 0.0
            
            return [in_deg, out_deg, clust, betw, pr, isolated]
        except:
            return [0.0] * 6
    
    def _transaction(self, user):
        """8 transaction features"""
        try:
            if user not in self.tx_history or not self.tx_history[user]:
                return [0.0] * 8
            
            txs = self.tx_history[user]
            amounts = [float(tx['amount']) for tx in txs if tx.get('amount', 0) > 0]
            
            if not amounts:
                return [0.0] * 8
            
            total = float(sum(amounts))
            avg = float(np.mean(amounts))
            std = float(np.std(amounts)) if len(amounts) > 1 else 0.0
            max_amt = float(max(amounts))
            count = float(len(txs))
            
            round_ratio = float(sum(1 for a in amounts if a % 100 == 0) / len(amounts))
            large_ratio = float(sum(1 for a in amounts if a > avg * 2) / len(amounts))
            
            if len(txs) > 1:
                try:
                    days = (txs[-1]['timestamp'] - txs[0]['timestamp']).days + 1
                    velocity = float(count / max(days, 1))
                except:
                    velocity = 0.0
            else:
                velocity = 0.0
            
            return [total, avg, std, max_amt, count, round_ratio, large_ratio, velocity]
        except:
            return [0.0] * 8
    
    def _temporal(self, user):
        """4 temporal features"""
        try:
            if user not in self.tx_history or not self.tx_history[user]:
                return [0.0] * 4
            
            txs = self.tx_history[user]
            
            night = sum(1 for tx in txs if 'timestamp' in tx and 0 <= tx['timestamp'].hour < 6)
            night_ratio = float(night / len(txs))
            
            weekend = sum(1 for tx in txs if 'timestamp' in tx and tx['timestamp'].weekday() >= 5)
            weekend_ratio = float(weekend / len(txs))
            
            if len(txs) > 1:
                try:
                    diffs = [(txs[i+1]['timestamp'] - txs[i]['timestamp']).total_seconds() / 3600
                            for i in range(len(txs)-1)
                            if 'timestamp' in txs[i] and 'timestamp' in txs[i+1]]
                    
                    if diffs:
                        burst_ratio = float(sum(1 for d in diffs if d < 1) / len(diffs))
                        avg_gap = float(np.mean(diffs))
                    else:
                        burst_ratio, avg_gap = 0.0, 0.0
                except:
                    burst_ratio, avg_gap = 0.0, 0.0
            else:
                burst_ratio, avg_gap = 0.0, 0.0
            
            return [night_ratio, weekend_ratio, burst_ratio, avg_gap]
        except:
            return [0.0] * 4
    
    def _anomaly(self, user):
        """3 anomaly features"""
        try:
            if user not in self.graph or user not in self.tx_history:
                return [0.0] * 3
            
            txs = self.tx_history[user]
            
            # Spike ratio
            if len(txs) >= 5:
                try:
                    recent = np.mean([float(tx['amount']) for tx in txs[-3:]])
                    historical = np.mean([float(tx['amount']) for tx in txs[:-3]])
                    spike = float(recent / historical) if historical > 0 else 1.0
                except:
                    spike = 1.0
            else:
                spike = 1.0
            
            # Balance ratio
            try:
                in_flow = sum(self.graph[p][user].get('weight', 0)
                            for p in self.graph.predecessors(user))
                out_flow = sum(self.graph[user][s].get('weight', 0)
                             for s in self.graph.successors(user))
                balance = float(abs(in_flow - out_flow) / max(in_flow + out_flow, 1))
            except:
                balance = 0.0
            
            # Counterparty ratio
            try:
                unique = len(set(tx.get('to', '') for tx in txs if 'to' in tx))
                counterparty = float(unique / len(txs))
            except:
                counterparty = 0.0
            
            return [spike, balance, counterparty]
        except:
            return [0.0] * 3

# GNN MODEL
class FraudGNN(torch.nn.Module):
    """Graph Attention Network for fraud detection"""
    
    def __init__(self, input_dim=21, hidden_dim=64, num_heads=4, dropout=0.3):
        super(FraudGNN, self).__init__()
        
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, data):
        try:
            x, edge_index = data.x, data.edge_index
            
            x1 = self.conv1(x, edge_index)
            x1 = F.elu(x1)
            if x1.size(0) > 1:
                x1 = self.bn1(x1)
            x1 = self.dropout(x1)
            
            x2 = self.conv2(x1, edge_index)
            x2 = F.elu(x2)
            if x2.size(0) > 1:
                x2 = self.bn2(x2)
            x2 = self.dropout(x2)
            
            x3 = self.conv3(x2, edge_index)
            x3 = F.elu(x3)
            x3 = x2 + x3
            
            out = self.classifier(x3)
            return F.log_softmax(out, dim=1)
        except Exception as e:
            print(f"Forward pass error: {e}")
            return torch.zeros(data.x.size(0), 2)


def main():
    """Main execution pipeline"""
    print("GNN-BASED FRAUD DETECTION SYSTEM")    
    # STAGE 1: Download dataset
    print("1) DATASET DOWNLOAD")
    downloader = DatasetDownloader()
    dataset_path = downloader.download_creditcard_dataset()
    
    # STAGE 2: Load data
    print("2) DATA LOADING")
    loader = RealDatasetLoader()
    graph, tx_history, labels = loader.load_creditcard_dataset(dataset_path)
    
    # STAGE 3: Extract features
    print("3) FEATURE EXTRACTION")
    extractor = FeatureExtractor(graph, tx_history)
    
    node_list = list(graph.nodes())
    print(f"\n Extracting 21 features for {len(node_list):,} nodes: ")
    
    features_dict = {}
    for i, node in enumerate(node_list):
        features_dict[node] = extractor.extract_all_features(node)
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i+1:,} / {len(node_list):,} nodes. ")
    
    print(f" Feature extraction complete!")
    
    # STAGE 4: Prepare PyTorch Geometric data
    print("4) PREPARING MODEL DATA")
    
    x = torch.tensor([features_dict[n] for n in node_list], dtype=torch.float)
    edge_index = torch.tensor(
        [[node_list.index(u), node_list.index(v)] for u, v in graph.edges()],
        dtype=torch.long
    ).t().contiguous()
    y = torch.tensor([labels.get(n, 0) for n in node_list], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    print(f" Data prepared:")
    print(f"   Nodes: {x.size(0):,}")
    print(f"   Features: {x.size(1)}")
    print(f"   Edges: {edge_index.size(1):,}")
    print(f"   Fraud: {y.sum().item():,} ({y.float().mean().item()*100:.2f}%)")
    
    # STAGE 5: Train model
    print("5) MODEL TRAINING")
    
    model = FraudGNN(input_dim=21, hidden_dim=64, num_heads=4)
    print(f" Model initialized")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # FIX 3: Calculate class weights from actual data - NO DEFAULTS, FAIL IF NO FRAUD
    fraud_count = (y == 1).sum().item()
    legit_count = (y == 0).sum().item()
    total = fraud_count + legit_count
    
    # Strict validation - MUST have fraud nodes for accurate training
    if fraud_count == 0:
        print(" CRITICAL ERROR: NO FRAUD NODES DETECTED!")
        print("The dataset processing failed to identify any fraudulent users.")
        print("This means the model cannot be trained accurately.")
        print("\nPossible causes:")
        print("1. All fraud transactions were distributed such that no user has fraud")
        print("2. Data clustering parameters need adjustment")
        print("3. Dataset corruption or incorrect file")
        print("\nCannot continue - training would produce meaningless results.")
        sys.exit(1)
    
    if legit_count == 0:
        print(" CRITICAL ERROR: NO LEGITIMATE NODES DETECTED!")
        print("The dataset processing marked all users as fraudulent.")
        print("This is unrealistic and means the model cannot be trained accurately.")
        print("\nCannot continue - training would produce meaningless results.")
        sys.exit(1)
    
    # Calculate accurate class weights
    weights = torch.tensor([
        total / (2.0 * legit_count),
        total / (2.0 * fraud_count)
    ])
    
    print(f"   Class weights: {weights[0]:.4f}, Fraud: {weights[1]:.4f}")
    
    print(f"\n Training for 50 epochs: ")
    model.train()
    
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, y, weight=weights)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print(" Training complete!")
    
    # STAGE 6: Evaluate
    print("6) EVALUATION")
    
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        
        correct = (pred == y).sum().item()
        accuracy = correct / len(y) * 100
        
        tp = ((pred == 1) & (y == 1)).sum().item()
        fp = ((pred == 1) & (y == 0)).sum().item()
        fn = ((pred == 0) & (y == 1)).sum().item()
        tn = ((pred == 0) & (y == 0)).sum().item()
        
        precision = tp / max(tp + fp, 1) * 100
        recall = tp / max(tp + fn, 1) * 100
        f1 = 2 * precision * recall / max(precision + recall, 1)
        
        print(f"\n RESULTS:")
        print(f"   Accuracy:  {accuracy:.2f}%")
        print(f"   Precision: {precision:.2f}%")
        print(f"   Recall:    {recall:.2f}%")
        print(f"   F1-Score:  {f1:.2f}%")
        print(f"\n  Confusion Matrix:")
        print(f"   True Positives:  {tp:,}")
        print(f"   False Positives: {fp:,}")
        print(f"   True Negatives:  {tn:,}")
        print(f"   False Negatives: {fn:,}")
    
    print(" SYSTEM RUN COMPLETE!")



if __name__ == '__main__':
    main()