"""
Graph Neural Network Module for Fixed Income RL Project

This module implements graph neural networks for modeling issuer relationships and credit risk:
1. Construction of bond issuer graphs
2. Node and edge feature engineering
3. GNN architecture for credit risk propagation
4. Training and inference utilities

Mathematical foundations:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Message Passing Neural Networks (MPNN)

Author: ranycs & cosrv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CreditRiskGNN:
    """
    Class for building and training GNNs for credit risk propagation.
    """
    
    def __init__(self, model_type: str = 'gcn', hidden_dim: int = 64, 
                num_layers: int = 2, device: str = 'cuda'):
        """
        Initialize the Credit Risk GNN.
        
        Args:
            model_type: Type of GNN ('gcn', 'gat', or 'mpnn')
            hidden_dim: Dimension of hidden layers
            num_layers: Number of message passing layers
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Placeholder for the model
        self.model = None
        
        # Data preprocessing
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        
        # Node and edge feature dimensions
        self.node_dim = None
        self.edge_dim = None
    
    def prepare_graph_data(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                        target_col: Optional[str] = None) -> Data:
        """
        Prepare graph data for GNN.
        
        Args:
            nodes_df: DataFrame with node features
            edges_df: DataFrame with edge features
            target_col: Column name for the target variable
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info(f"Preparing graph data with {len(nodes_df)} nodes and {len(edges_df)} edges")
        
        # Check input data
        logger.info(f"Nodes DataFrame index type: {type(nodes_df.index[0])}")
        logger.info(f"First 5 node indices: {list(nodes_df.index[:5])}")
        
        logger.info(f"Edges DataFrame source column type: {type(edges_df['source'].iloc[0]) if not edges_df.empty else 'N/A'}")
        logger.info(f"First 5 source values: {list(edges_df['source'].iloc[:5]) if len(edges_df) >= 5 else list(edges_df['source']) if not edges_df.empty else []}")
        
        # Convert node indices and edge source/target to string if they're different types
        original_index = nodes_df.index
        if not edges_df.empty and not isinstance(edges_df['source'].iloc[0], type(nodes_df.index[0])):
            logger.warning(f"Type mismatch between nodes index ({type(nodes_df.index[0])}) and edges source ({type(edges_df['source'].iloc[0])})")
            
            # Convert both to string for consistent mapping
            nodes_df = nodes_df.reset_index()
            nodes_df['original_index'] = nodes_df['index'].astype(str)
            nodes_df = nodes_df.set_index('original_index')
            
            # Convert edge source/target to string
            edges_df['source'] = edges_df['source'].astype(str)
            edges_df['target'] = edges_df['target'].astype(str)
            
            logger.info("Converted indices to strings for consistent mapping")
            logger.info(f"Updated node index type: {type(nodes_df.index[0])}")
            logger.info(f"Updated edge source type: {type(edges_df['source'].iloc[0])}")
        
        # Create node mapping (node ID to index)
        node_map = {node_id: idx for idx, node_id in enumerate(nodes_df.index)}
        logger.info(f"Created node mapping with {len(node_map)} entries")
        
        # Sample a few entries from the node map
        sample_entries = list(node_map.items())[:5]
        logger.info(f"Sample node map entries: {sample_entries}")
        
        # Process node features
        categorical_cols = nodes_df.select_dtypes(include=['object', 'category']).columns
        
        # One-hot encode categorical features
        node_features_df = pd.get_dummies(nodes_df, columns=categorical_cols)
        
        # Scale numerical features
        node_features = node_features_df.values
        self.node_dim = node_features.shape[1]
        
        # Fit scaler if needed
        if not hasattr(self.node_scaler, 'mean_'):
            self.node_scaler.fit(node_features)
        
        # Transform features
        node_features = self.node_scaler.transform(node_features)
        
        # Process edges
        edge_list = []
        edge_weights = []
        
        for idx, edge in edges_df.iterrows():
            source = edge['source']
            target = edge['target']
            
            # Map to indices
            source_idx = node_map.get(source)
            target_idx = node_map.get(target)
            
            if source_idx is not None and target_idx is not None:
                edge_list.append([source_idx, target_idx])
                # Also add the reverse edge for undirected graph
                edge_list.append([target_idx, source_idx])
                
                # Add edge weights if available
                if 'weight' in edge.index:
                    edge_weights.append(edge['weight'])
                    edge_weights.append(edge['weight'])  # Same weight for reverse edge
            else:
                if source_idx is None:
                    logger.warning(f"Node ID {source} not found in node map")
                if target_idx is None:
                    logger.warning(f"Node ID {target} not found in node map")
        
        # Log mapping stats
        mapped_sources = sum(1 for edge in edges_df['source'] if edge in node_map)
        mapped_targets = sum(1 for edge in edges_df['target'] if edge in node_map)
        logger.info(f"Successfully mapped source nodes: {mapped_sources}/{len(edges_df)} ({mapped_sources/len(edges_df)*100:.1f}%)")
        logger.info(f"Successfully mapped target nodes: {mapped_targets}/{len(edges_df)} ({mapped_targets/len(edges_df)*100:.1f}%)")
        
        # Check if we have edges before converting
        if len(edge_list) == 0:
            logger.warning("No valid edges found after mapping node IDs to indices!")
            # Create a default edge to prevent errors (will be removed later if needed)
            edge_list.append([0, 0])
            if len(edge_weights) > 0:
                edge_weights.append(edge_weights[0])
        else:
            logger.info(f"Successfully mapped {len(edge_list)//2} edges (with reverse edges: {len(edge_list)})")
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Process edge features if available
        if 'weight' in edges_df.columns and len(edge_weights) > 0:
            edge_attr = np.array(edge_weights)
            
            # Reshape for scaling
            edge_attr = edge_attr.reshape(-1, 1)
            
            # Fit scaler if needed
            if not hasattr(self.edge_scaler, 'mean_'):
                self.edge_scaler.fit(edge_attr)
            
            # Transform features
            edge_attr = self.edge_scaler.transform(edge_attr)
            
            # Convert to PyTorch tensor
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            self.edge_dim = edge_attr.shape[1]
        else:
            edge_attr = None
            self.edge_dim = 0
        
        # Process target variable if available
        if target_col and target_col in nodes_df.columns:
            y = torch.tensor(nodes_df[target_col].values, dtype=torch.float)
        else:
            y = None
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )
        
        # Log diagnostic information
        logger.info(f"Prepared graph data with {data.num_nodes} nodes and {data.num_edges} edges")
        if data.num_edges <= 1:  # Just the dummy edge
            logger.error("No real edges in the prepared graph data! Check edge processing logic.")
            logger.debug(f"Edge list before conversion: {len(edge_list)} entries")
            logger.debug(f"First few edges: {edge_list[:5] if len(edge_list) > 5 else edge_list}")
            
        return data
    
    def build_model(self):
        """
        Build the GNN model.
        """
        logger.info(f"Building {self.model_type} model")
        
        if self.node_dim is None:
            raise ValueError("Node feature dimension not set. Call prepare_graph_data() first.")
        
        if self.model_type == 'gcn':
            self.model = GCNModel(
                input_dim=self.node_dim,
                hidden_dim=self.hidden_dim,
                output_dim=1,
                num_layers=self.num_layers,
                edge_dim=self.edge_dim
            ).to(self.device)
            
        elif self.model_type == 'gat':
            self.model = GATModel(
                input_dim=self.node_dim,
                hidden_dim=self.hidden_dim,
                output_dim=1,
                num_layers=self.num_layers,
                edge_dim=self.edge_dim,
                num_heads=4
            ).to(self.device)
            
        elif self.model_type == 'mpnn':
            self.model = MPNNModel(
                input_dim=self.node_dim,
                hidden_dim=self.hidden_dim,
                output_dim=1,
                num_layers=self.num_layers,
                edge_dim=self.edge_dim
            ).to(self.device)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Built {self.model_type} model with {self.hidden_dim} hidden dimensions and {self.num_layers} layers")
        
        return self.model
    
    def train(self, data: Data, epochs: int = 100, learning_rate: float = 0.01, 
            weight_decay: float = 5e-4, verbose: bool = True) -> List[float]:
        """
        Train the GNN model.
        
        Args:
            data: PyTorch Geometric Data object
            epochs: Number of epochs to train
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            verbose: Whether to print progress
            
        Returns:
            List of loss values
        """
        if self.model is None:
            self.build_model()
        
        # Move data to device
        data = data.to(self.device)
        
        # Check if target variable is available
        if data.y is None:
            raise ValueError("Target variable not found in data")
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Train the model
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            
            # Calculate loss
            loss = criterion(output, data.y.view(-1, 1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Store loss
            losses.append(loss.item())
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        logger.info(f"Training completed with final loss: {losses[-1]:.4f}")
        
        return losses
    
    def predict(self, data: Data) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Move data to device
        data = data.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            output = self.model(data)
            predictions = output.cpu().numpy()
        
        return predictions
    
    def evaluate(self, data: Data) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Check if target variable is available
        if data.y is None:
            raise ValueError("Target variable not found in data")
        
        # Make predictions
        predictions = self.predict(data)
        
        # Calculate metrics
        mse = np.mean((predictions - data.y.cpu().numpy().reshape(-1, 1)) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - data.y.cpu().numpy().reshape(-1, 1)))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'node_scaler': self.node_scaler,
            'edge_scaler': self.edge_scaler
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Extract model parameters
        self.model_type = checkpoint['model_type']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.node_dim = checkpoint['node_dim']
        self.edge_dim = checkpoint['edge_dim']
        self.node_scaler = checkpoint['node_scaler']
        self.edge_scaler = checkpoint['edge_scaler']
        
        # Build model
        self.build_model()
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_graph(self, data: Data, node_labels: Optional[List[str]] = None,
                 predictions: Optional[np.ndarray] = None, 
                 figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot the graph using NetworkX.
        
        Args:
            data: PyTorch Geometric Data object
            node_labels: List of node labels
            predictions: Node predictions
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Check if data object has edges
        if data.edge_index is None or data.num_edges == 0:
            logger.warning("No edges in the graph for visualization. Creating a placeholder visualization.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No edges available for graph visualization!", 
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title("Credit Risk Graph - ERROR: No Edges")
            plt.axis('off')
            return fig
            
        logger.info(f"Plotting graph with {data.num_nodes} nodes and {data.num_edges} edges")
        
        # Convert to NetworkX graph
        G = self._to_networkx(data)
        
        # If the graph is still empty, create a placeholder visualization
        if G.number_of_edges() == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Graph conversion resulted in no edges!", 
                   horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title("Credit Risk Graph - ERROR: No Edges After Conversion")
            plt.axis('off')
            return fig
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set node colors based on predictions
        if predictions is not None:
            node_colors = predictions
        else:
            node_colors = 'skyblue'
            
        # Set node sizes based on degree
        node_sizes = [300 * (1 + G.degree(node) / 10) for node in G.nodes()]
        
        # Create a spring layout for the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            cmap=plt.cm.viridis,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos,
            width=1.0,
            alpha=0.5,
            ax=ax
        )
        
        # Draw labels if provided
        if node_labels is not None:
            # Ensure correct length
            if len(node_labels) != data.num_nodes:
                logger.warning(f"Number of labels ({len(node_labels)}) does not match number of nodes ({data.num_nodes})")
                node_labels = node_labels[:data.num_nodes]
            
            # Create label dictionary
            labels = {i: label for i, label in enumerate(node_labels)}
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_size=8
            )
        
        # Add colorbar if using node colors
        if isinstance(node_colors, np.ndarray):
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max()))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.7)
        
        # Print some graph statistics
        graph_info = f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
        ax.text(0.05, 0.05, graph_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Set title and layout
        plt.title("Credit Risk Graph")
        plt.axis('off')
        plt.tight_layout()
        
        return fig
    
    def _to_networkx(self, data: Data) -> nx.Graph:
        """
        Convert PyTorch Geometric Data to NetworkX graph.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i)
        
        # Add edges
        if data.edge_index is None or data.edge_index.shape[1] == 0:
            logger.warning("No edges in the PyTorch Geometric Data object!")
            return G
            
        edge_index = data.edge_index.cpu().numpy()
        logger.info(f"Converting {edge_index.shape[1]} edges from PyG to NetworkX")
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            
            # Add edge attributes if available
            if data.edge_attr is not None:
                weight = data.edge_attr[i].item()
                G.add_edge(src.item(), dst.item(), weight=weight)
            else:
                G.add_edge(src.item(), dst.item())
        
        logger.info(f"NetworkX graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def generate_node_embeddings(self, data: Data) -> np.ndarray:
        """
        Generate node embeddings using the trained model.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Array of node embeddings
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Move data to device
        data = data.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.get_embeddings(data)
            embeddings = embeddings.cpu().numpy()
        
        return embeddings


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) model.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                num_layers: int, edge_dim: int = 0, dropout: float = 0.5):
        """
        Initialize the GCN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_layers: Number of GCN layers
            edge_dim: Dimension of edge features (if any)
            dropout: Dropout probability
        """
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Model output
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply convolutional layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Apply final MLP
        x = self.mlp(x)
        
        return x
    
    def get_embeddings(self, data):
        """
        Get node embeddings before the final MLP.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node embeddings
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply convolutional layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) model.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                num_layers: int, edge_dim: int = 0, num_heads: int = 4, 
                dropout: float = 0.5):
        """
        Initialize the GAT model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_layers: Number of GAT layers
            edge_dim: Dimension of edge features (if any)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer (input dim -> hidden dim * num_heads)
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout, edge_dim=edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final MLP
        final_dim = hidden_dim if num_layers > 1 else hidden_dim * num_heads
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Model output
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply convolutional layers
        for i in range(self.num_layers):
            if edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Apply final MLP
        x = self.mlp(x)
        
        return x
    
    def get_embeddings(self, data):
        """
        Get node embeddings before the final MLP.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node embeddings
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Apply convolutional layers
        for i in range(self.num_layers):
            if edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x


class EdgeConv(MessagePassing):
    """
    Edge convolutional layer for Message Passing Neural Network.
    """
    
    def __init__(self, in_channels, out_channels, edge_channels=0):
        """
        Initialize the EdgeConv layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            edge_channels: Number of edge feature channels
        """
        super(EdgeConv, self).__init__(aggr='max')
        
        # MLP for node features
        self.mlp_x = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_attr: Edge features
            
        Returns:
            Updated node features
        """
        # Propagate messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """
        Message function.
        
        Args:
            x_i: Features of target nodes
            x_j: Features of source nodes
            edge_attr: Edge features
            
        Returns:
            Messages
        """
        # Concatenate source and target node features
        msg = torch.cat([x_i, x_j], dim=1)
        
        # Add edge features if available
        if edge_attr is not None:
            msg = torch.cat([msg, edge_attr], dim=1)
        
        # Apply MLP
        return self.mlp_x(msg)


class MPNNModel(nn.Module):
    """
    Message Passing Neural Network (MPNN) model.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                num_layers: int, edge_dim: int = 0, dropout: float = 0.5):
        """
        Initialize the MPNN model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_layers: Number of message passing layers
            edge_dim: Dimension of edge features (if any)
            dropout: Dropout probability
        """
        super(MPNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Message passing layers
        for _ in range(num_layers):
            self.convs.append(EdgeConv(hidden_dim, hidden_dim, edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Model output
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Apply message passing layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Apply final MLP
        x = self.mlp(x)
        
        return x
    
    def get_embeddings(self, data):
        """
        Get node embeddings before the final MLP.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Node embeddings
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Apply message passing layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
