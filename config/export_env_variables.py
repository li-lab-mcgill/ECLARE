import os
import subprocess
import yaml
import sys


def load_config(config_file):
    """Load the YAML configuration file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main(config_file):
    config = load_config(config_file)

    # Extract the active environment
    active_env = config.get('active_environment')
    if not active_env:
        print("Error: 'active_environment' not found in the configuration file.")
        sys.exit(1)

    # Extract values based on the active environment
    env_config = config.get(active_env, {})
    eclare_root = env_config.get('ECLARE_ROOT')
    outpath = env_config.get('OUTPATH')
    datapath = env_config.get('DATAPATH')

    if not all([eclare_root, outpath, datapath]):
        print(f"Error: Missing environment variables for {active_env}.")
        sys.exit(1)

    # Set the environment variables
    os.environ['ECLARE_ROOT'] = eclare_root
    os.environ['OUTPATH'] = outpath
    os.environ['DATAPATH'] = datapath

    export_commands = f"""export ECLARE_ROOT={eclare_root}
export OUTPATH={outpath}
export DATAPATH={datapath}"""
    print(export_commands)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python export_env_variables.py <config_file>")
        sys.exit(1)

    main(sys.argv[1])
