"""
ModelExporter.py

Exports trained PyTorch model to binary format compatible with C# wrapper.
This script converts your trained model to a format that DWSIMNeuralNetworkWrapper.cs can load.
"""

import torch
import numpy as np
import joblib
import struct
import json
import os
from pathlib import Path


class ModelExporter:
    """Export PyTorch model to C#-compatible binary format"""
    
    def __init__(self, model, scaler_x, scaler_y):
        """
        Parameters:
            model: PyTorch neural network model
            scaler_x: sklearn StandardScaler for inputs
            scaler_y: sklearn StandardScaler for outputs
        """
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
    
    def export(self, output_dir="exported_model", config_file="config.json"):
        """
        Export model to binary format
        
        Parameters:
            output_dir: Directory to save exported files
            config_file: Configuration JSON file name
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export weights and biases
        model_file = os.path.join(output_dir, "model.dat")
        self._export_model_binary(model_file)
        
        # Export scalers
        scaler_file = os.path.join(output_dir, "scalers.dat")
        self._export_scalers_binary(scaler_file)
        
        # Create configuration file
        config_file_path = os.path.join(output_dir, config_file)
        self._create_config_file(config_file_path, "model.dat", "scalers.dat")
        
        print(f"✓ Model exported to {output_dir}")
        print(f"  - Model weights: {model_file}")
        print(f"  - Scalers: {scaler_file}")
        print(f"  - Config: {config_file_path}")
        
        return output_dir
    
    def _export_model_binary(self, filepath):
        """Export PyTorch model weights and biases to binary format"""
        
        with open(filepath, 'wb') as f:
            # Collect all linear layers
            linear_layers = [m for m in self.model.modules() 
                           if isinstance(m, torch.nn.Linear)]
            
            # Write number of layers
            f.write(struct.pack('i', len(linear_layers)))
            
            # Write each layer's weights and biases
            for layer in linear_layers:
                # Get weights and biases
                weights = layer.weight.data.cpu().numpy()  # shape: (out_features, in_features)
                bias = layer.bias.data.cpu().numpy()        # shape: (out_features,)
                
                # Write weight matrix dimensions
                f.write(struct.pack('i', weights.shape[0]))  # output size
                f.write(struct.pack('i', weights.shape[1]))  # input size
                
                # Write weights (flattened row-major)
                for val in weights.flatten():
                    f.write(struct.pack('d', float(val)))
                
                # Write bias vector
                for val in bias:
                    f.write(struct.pack('d', float(val)))
    
    def _export_scalers_binary(self, filepath):
        """Export sklearn scalers to binary format"""
        
        with open(filepath, 'wb') as f:
            # Input scaler (mean and scale)
            x_mean = self.scaler_x.mean_
            x_scale = self.scaler_x.scale_
            
            # Write input scaler size and values
            f.write(struct.pack('i', len(x_mean)))
            for val in x_mean:
                f.write(struct.pack('d', float(val)))
            for val in x_scale:
                f.write(struct.pack('d', float(val)))
            
            # Output scaler (mean and scale)
            y_mean = self.scaler_y.mean_
            y_scale = self.scaler_y.scale_
            
            # Write output scaler size and values
            f.write(struct.pack('i', len(y_mean)))
            for val in y_mean:
                f.write(struct.pack('d', float(val)))
            for val in y_scale:
                f.write(struct.pack('d', float(val)))
    
    def _create_config_file(self, filepath, model_file, scaler_file):
        """Create JSON configuration file"""
        
        # Get model dimensions from first and last linear layers
        linear_layers = [m for m in self.model.modules() 
                        if isinstance(m, torch.nn.Linear)]
        
        input_count = linear_layers[0].in_features
        output_count = linear_layers[-1].out_features
        
        config = {
            "name": "Custom DWSIM ANN Model",
            "input_count": input_count,
            "output_count": output_count,
            "activation": "tanh",  # Change if using different activation
            "model_file": model_file,
            "scaler_file": scaler_file,
            "hidden_layers": [m.out_features for m in linear_layers[:-1]]
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


def export_trained_model(model_path, scaler_x_path, scaler_y_path, output_dir="exported_model"):
    """
    Convenience function to export a trained model
    
    Usage:
        export_trained_model(
            model_path="dwsim_ann.pt",
            scaler_x_path="scaler_x.gz",
            scaler_y_path="scaler_y.gz",
            output_dir="exported_model"
        )
    """
    
    # Load PyTorch model
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dwsim_like_ann import DWSIMLikeMLP
    
    # Determine dimensions from saved model
    checkpoint = torch.load(model_path, map_location='cpu')
    first_weight_shape = checkpoint['net.0.weight'].shape
    input_dim = first_weight_shape[1]
    
    # Count output dimension
    last_key = [k for k in checkpoint.keys() if 'weight' in k][-1]
    output_dim = checkpoint[last_key].shape[0]
    
    model = DWSIMLikeMLP(
        n_in=input_dim,
        n_out=output_dim,
        hidden_sizes=[100],  # Adjust based on your model
        activation="tanh"
    )
    model.load_state_dict(checkpoint)
    
    # Load scalers
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # Export
    exporter = ModelExporter(model, scaler_x, scaler_y)
    exporter.export(output_dir)


if __name__ == "__main__":
    # Example usage
    export_trained_model(
        model_path="dwsim_ann.pt",
        scaler_x_path="scaler_x.gz",
        scaler_y_path="scaler_y.gz",
        output_dir="exported_model"
    )
