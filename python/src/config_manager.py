import json
import os
from pathlib import Path

class CactusConfig:
    def __init__(self):
        self.config_dir = Path.home() / ".cactus"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)

    def load_config(self):
        if self.config_file.exists():
            return json.loads(self.config_file.read_text())
        return {}

    def save_config(self, config):
        self.config_file.write_text(json.dumps(config, indent=2))

    def get_api_key(self):
        env_key = os.getenv("CACTUS_CLOUD_API_KEY")
        if env_key:
            return env_key
        config = self.load_config()
        return config.get("api_key", "")

    def set_api_key(self, key):
        config = self.load_config()
        config["api_key"] = key
        self.save_config(config)
