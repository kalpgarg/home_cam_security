#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
echo "The repo location is: $REPO_DIR"

echo "Creating virtual environment"
source REPO_DIR/home_cam/bin/activate

echo "Running script for fetching from camera no 1"
output=$(python REPO_DIR/fetch_save_stream.py --cl REPO_DIR/cp_plus_cam_info.json &)
echo "$output" > REPO_DIR/log/fetch_streams.txt
echo "fetch_save_stream is running in the background!"
echo "Output is logged in location: $REPO_DIR/log/fetch_streams.txt"

