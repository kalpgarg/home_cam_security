#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR=$(dirname "$SCRIPT_DIR")
echo "The repo location is: $REPO_DIR"

echo "Creating virtual environment"
source $REPO_DIR/home_cam/bin/activate

OUTPUT_LOG=$REPO_DIR/results/fetch_streams.txt
RECORDINGS_DIR=$REPO_DIR/cam_stream_log
echo "Running script for fetching from camera no 1"
python $REPO_DIR/fetch_save_stream.py -cl $REPO_DIR/custom_cam_info.json -l $RECORDINGS_DIR -cn 1 >> $OUTPUT_LOG 2>&1 &
echo "Running script for fetching from camera no 2"
python $REPO_DIR/fetch_save_stream.py -cl $REPO_DIR/custom_cam_info.json -l $RECORDINGS_DIR -cn 2 >> $OUTPUT_LOG 2>&1 &
echo "fetch_save_stream is running in the background!"
echo "Output is logged in location: $OUTPUT_LOG"

