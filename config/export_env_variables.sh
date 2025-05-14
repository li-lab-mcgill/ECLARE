#!/bin/bash

echo '''
Exporting environment variables from config.yaml - script needs to be sourced and path to config.yaml needs to be provided as argument. For example:

>> source config/export_env_variables.sh config/config.yaml
'''

# Use Python to generate the export commands and capture them
export_commands=$(python -c """
import os
import yaml
import sys
import re
import socket

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main(config_file):
    config = load_config(config_file)

    # Get the hostname mapping and current hostname
    hostname_mapping = config.get('hostname_mapping', {})
    hostname = socket.gethostname()

    # Try to match the hostname (exact or regex)
    active_env = None
    for pattern, env in hostname_mapping.items():
        if pattern.startswith('/') and pattern.endswith('/'):
            # Regex pattern
            regex = pattern[1:-1]
            if re.match(regex, hostname):
                active_env = env
                break
        else:
            # Exact match
            if pattern == hostname:
                active_env = env
                break

    # Fallback to default active_environment if no match
    if not active_env:
        active_env = config.get('active_environment')

    # Extract values based on the active environment
    env_config = config.get(active_env, {})
    eclare_root = env_config.get('ECLARE_ROOT')
    outpath = env_config.get('OUTPATH')
    datapath = env_config.get('DATAPATH')

    # Prepare the export commands
    export_commands = f'''export ACTIVE_ENV={active_env}
export ECLARE_ROOT={eclare_root}
export OUTPATH={outpath}
export DATAPATH={datapath}'''
    print(export_commands)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python export_env_variables.py <config_file>')
        sys.exit(1)

    main(sys.argv[1])
""" "$1")

# Evaluate the export commands to set the environment variables
eval "$export_commands"

echo "Environment variables set for $ACTIVE_ENV:"
echo -e "\t - ECLARE_ROOT=$ECLARE_ROOT"
echo -e "\t - OUTPATH=$OUTPATH"
echo -e "\t - DATAPATH=$DATAPATH"
echo -e '\n'