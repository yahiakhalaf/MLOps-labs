import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load and validate configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert relative paths to absolute
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path_key in ["model_output", "metrics_output", "log_file"]:
        if path_key in config.get("paths", {}):
            config["paths"][path_key] = os.path.join(base_dir, config["paths"][path_key])
    
    return config