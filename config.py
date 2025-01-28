import os
from pathlib import Path
import yaml

class Config:
    def __init__(self, config_file: Path = Path("config.yaml")):
        self.config_file = config_file
        self.directories = {}
        self.load_config()

    def load_config(self):
        # Load configuration from YAML file
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                self.directories = config.get('directories', {})
        else:
            print(f"Configuration file {self.config_file} not found. Using defaults or environment variables.")

        # Override with environment variables if set
        self.directories['data_dir'] = Path(os.getenv('DATA_DIR', self.directories.get('data_dir', './data')))
        self.directories['logs_dir'] = Path(os.getenv('LOGS_DIR', self.directories.get('logs_dir', './logs')))
        self.directories['output_dir'] = Path(os.getenv('OUTPUT_DIR', self.directories.get('output_dir', './output')))

    def get_data_dir(self) -> Path:
        return self.directories['data_dir']

    def get_logs_dir(self) -> Path:
        return self.directories['logs_dir']

    def get_output_dir(self) -> Path:
        return self.directories['output_dir']

# Instantiate a global config object
config = Config()
