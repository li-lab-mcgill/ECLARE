# config_loader.py

import os
import socket
from pathlib import Path
import yaml
from typing import Any, Dict, Optional
import re
import argparse

class ConfigLoader:
    def __init__(self, config_file: Path = Path("config.yaml")):
        self.config_file = config_file
        self.config_data = self.load_yaml()
        self.hostname_mapping = self.config_data.get('hostname_mapping', {})
        self.active_env = self.determine_active_environment()
        self.directories = self.get_directories()

    def load_yaml(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")

        with open(self.config_file, 'r') as f:
            try:
                config = yaml.safe_load(f)
                if not config:
                    raise ValueError("Configuration file is empty.")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

        return config

    def determine_active_environment(self) -> str:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Select configuration environment.")
        parser.add_argument('--env', type=str, help='Active environment (e.g., local_directories, narval_directories, mcb_directories)')
        args, unknown = parser.parse_known_args()

        # Detect environment based on hostname
        hostname = socket.gethostname()
        print(f"Detected hostname: {hostname}")

        active_env = self.detect_environment(hostname, self.hostname_mapping)

        if active_env:
            print(f"Active environment detected based on hostname: {active_env}")
            return active_env
        else:
            # Check if '--env' argument was provided
            if args.env:
                if args.env in self.config_data:
                    print(f"Using environment specified via command-line argument: {args.env}")
                    self.save_active_environment(args.env)
                    return args.env
                else:
                    raise ValueError(f"Environment '{args.env}' not found in config.yaml.")
            # Check if 'active_environment' is set in config.yaml
            active_env = self.config_data.get('active_environment')
            if active_env and active_env in self.config_data:
                print(f"Using 'active_environment' from config.yaml: {active_env}")
                return active_env
            else:
                # Prompt user to select environment
                return self.prompt_user_for_environment()

    def detect_environment(self, hostname: str, mapping: Dict[str, str]) -> Optional[str]:
        """
        Detect the environment based on the hostname using exact matches or regex patterns.
        """
        for pattern, env in mapping.items():
            if pattern.startswith('/') and pattern.endswith('/'):
                # Regex pattern
                regex = pattern.strip('/')
                if re.match(regex, hostname):
                    return env
            else:
                # Exact match (case-insensitive)
                if pattern.lower() == hostname.lower():
                    return env
        return None

    def prompt_user_for_environment(self) -> str:
        """
        Prompt the user to select an active environment from available options.
        """
        available_envs = [key for key in self.config_data.keys() if key.endswith('_directories')]

        if not available_envs:
            raise ValueError("No directory configurations found in config.yaml.")

        print("\nNo active environment detected based on hostname or config.yaml.")
        print("Please select an environment from the following options:")

        for idx, env in enumerate(available_envs, start=1):
            print(f"{idx}. {env}")

        while True:
            try:
                choice = int(input(f"Enter the number corresponding to your choice (1-{len(available_envs)}): "))
                if 1 <= choice <= len(available_envs):
                    selected_env = available_envs[choice - 1]
                    print(f"Selected environment: {selected_env}")
                    # Optionally, save this selection to config.yaml
                    self.save_active_environment(selected_env)
                    return selected_env
                else:
                    print(f"Please enter a number between 1 and {len(available_envs)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    def save_active_environment(self, selected_env: str):
        """
        Save the selected active environment to config.yaml.
        """
        self.config_data['active_environment'] = selected_env
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config_data, f)
            print(f"Saved '{selected_env}' as the active environment in {self.config_file}.")
        except Exception as e:
            print(f"Warning: Failed to save active environment to {self.config_file}: {e}")

    def get_directories(self) -> Dict[str, Path]:
        """
        Retrieve directory paths for the active environment, applying any environment variable overrides.
        """
        directories = self.config_data.get(self.active_env, {})
        if not directories:
            raise ValueError(f"No directories found for environment '{self.active_env}'.")

        # Override with environment variables if set
        directories = self.override_with_env(directories)

        # Convert all paths to absolute Path objects
        for key, path in directories.items():
            directories[key] = Path(path).expanduser().resolve()

        # Validate and create directories if they don't exist
        self.validate_and_create_directories(directories)

        return directories

    def override_with_env(self, directories: Dict[str, str]) -> Dict[str, str]:
        """
        Override directory paths with environment variables if they are set.
        Environment variables should be in uppercase and match the directory keys.
        Example: ECLARE_ROOT overrides 'ECLARE_root'
        """
        env_mapping = {
            'ECLARE_root': 'ECLARE_ROOT',
            'outpath': 'OUTPATH',
            'datapath': 'DATAPATH'
        }

        for key, env_var in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value:
                directories[key] = env_value
                print(f"Overriding '{key}' with environment variable '{env_var}': {env_value}")

        return directories

    def validate_and_create_directories(self, directories: Dict[str, Path]):
        """
        Ensure that all directories exist. Create them if they don't.
        """
        for key, path in directories.items():
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"Created missing directory for '{key}': {path}")
                except Exception as e:
                    raise ValueError(f"Failed to create directory '{path}' for '{key}': {e}")

    def get_directory(self, key: str) -> Path:
        """
        Retrieve a specific directory path by key.
        """
        directory = self.directories.get(key)
        if not directory:
            raise KeyError(f"Directory '{key}' not found in the active configuration.")
        return directory

# Instantiate a global config object
#config_loader = ConfigLoader()
