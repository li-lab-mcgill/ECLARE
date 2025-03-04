import os
import subprocess

def export_env_variables(config_path='config'):
    # Ensure that the environment variables set by the script are accessible in this notebook

    # Run the export_env_variables.sh script and capture the output
    result = subprocess.run(['bash', '-c', f'{config_path}/export_env_variables.sh {config_path}/config.yaml'], capture_output=True, text=True)

    # Parse the output and set the environment variables in the current Python environment
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value

    # Verify that the environment variables are set
    print("ECLARE_ROOT:", os.environ.get("ECLARE_ROOT"))
    print("OUTPATH:", os.environ.get("OUTPATH"))
    print("DATAPATH:", os.environ.get("DATAPATH"))