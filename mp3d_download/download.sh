#!/bin/bash

# Set your paths
download_mp_script="/home/ubuntu/datasets/zubair/download_mp.py"
scans_txt_path="/home/ubuntu/datasets/zubair/scans.txt"
output_dir="/home/ubuntu/datasets/zubair"

# Create the output directory if it doesn't exist
data_path="$output_dir"
mkdir -p "$data_path"

# Check if the scans.txt file exists
if [ ! -f "$scans_txt_path" ]; then
  echo "Error: The scans.txt file could not be found. Check the argument."
  exit 1
fi

# Read the scan IDs from scans.txt
current_ids=()
while IFS= read -r id_val || [[ -n "$id_val" ]]; do
  if [ -n "$id_val" ]; then
    current_ids+=("$id_val")
  fi
done < "$scans_txt_path"

# Check if the download_mp script exists
download_mp_file="$download_mp_script"
if [ ! -f "$download_mp_file" ]; then
  echo "Error: The download_mp script could not be found: $download_mp_file"
  exit 1
fi

# Download Matterport Mesh for each scan ID
for current_id in "${current_ids[@]}"; do
  # The script only works with Python 2, and it only downloads the matterport_mesh
  cmd="python2 -u $download_mp_file -o $data_path --id $current_id --type matterport_mesh"
  agree="agree"

  # Run the command with subprocess
  echo "Running: $cmd"
  echo -e "$agree" | $cmd
done

# Extract and remove zip files
for zip_file in "$data_path/v1/scans/"*/*.zip; do
  unzip -q -d "$(dirname "$zip_file")" "$zip_file"
  rm "$zip_file"
done

echo "Script completed."
