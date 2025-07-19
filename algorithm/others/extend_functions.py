import yaml
import os



def extend_loadConfig(config_path="config/defaultImage_value.yaml") -> dict:
        
    default_config = {
        "image": {
            "ela": {
                "enabled": True,
                "compression": 85,
                "multiplier": 40
            },
            "noise": {
                "enabled": True,
                "local_mean": 7,
                "variance": 40
            },
            "copyMove": {
                "enabled": True,
                "block_size": 16,
                "step": 8,
                "threshold": 1000
            }
        }
    }


    os.makedirs(os.path.dirname(config_path), exist_ok=True) 
    if not os.path.isfile(config_path):
        with open(config_path, "w") as f:
            yaml.dump(default_config, f)
        print(f"[INFO] Created default config at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config




