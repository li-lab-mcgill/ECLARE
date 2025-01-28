import os
import requests
from bs4 import BeautifulSoup
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download files from a URL directory.')
parser.add_argument('--base_url', type=str, help='The base URL of the directory to download files from.')
parser.add_argument('--download_dir', type=str, default='/home/dmannk/projects/def-liyue/dmannk/data', help='The directory to save the downloaded files.')
args = parser.parse_args()

download_dir_basename = '_'.join(args.base_url.split('/'))
download_dir = os.path.join(args.download_dir, download_dir_basename)
base_url = args.base_url

# Create a directory to save the downloaded files
os.makedirs(download_dir, exist_ok=True)

# Fetch the HTML content of the directory listing
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Loop through all the <a> tags to find files
for link in soup.find_all('a'):
    file_name = link.get('href')
    if file_name and not file_name.startswith('/'):  # Ignore parent directory links
        file_url = base_url + file_name
        file_path = os.path.join(download_dir, file_name)
        print(f"Downloading {file_url} to {file_path}")

        # Download the file
        file_response = requests.get(file_url)
        with open(file_path, 'wb') as file:
            file.write(file_response.content)

## Create a REAMDE file that includes the base URL
with open(os.path.join(download_dir, 'README.md'), 'w') as file:
    file.write(f"Downloaded from: {base_url}")

print("Download complete.")