#!/bin/bash
# This script downloads the C3VD dataset from the JHU Dataverse using the DOI and filename provided.
DOI="10.7281/T1/JC64MK"

# Ensure the filename matches the one in the dataset exactly, including case and extension.
filename="c1_cecum_t1_v1.zip"

curl "https://archive.data.jhu.edu/api/datasets/:persistentId?persistentId=doi:$DOI" | jq ".data.latestVersion.files[] | select(.label==\"$filename\") | .dataFile.id" | while read -r file_id; do
    echo "Downloading $filename"
    download_url=$(curl -s -D - -o /dev/null "https://archive.data.jhu.edu/api/access/datafile/$file_id" | grep -i '^Location: ' | cut -d' ' -f2)
    echo "Download URL: $download_url"
    curl -L -o "$filename" "$(echo "$download_url" | tr -d '\r')"
done

# Check if the file was downloaded
if [ -f "$filename" ]; then
    echo "File $filename downloaded successfully."
else
    echo "Failed to download $filename."
fi