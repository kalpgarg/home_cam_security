#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR=$(dirname "$SCRIPT_DIR")
echo "The repo location is: $REPO_DIR"

echo "Creating virtual environment"
source $REPO_DIR/home_cam/bin/activate

OUTPUT_LOG=$REPO_DIR/results/fetch_streams.txt
RECORDINGS_DIR=$REPO_DIR/cam_stream_log

# List of default cameras to run if no argument given
DEFAULT_CAMERAS=(1 2 4)

# Camera number to run (optional)
CAM_TO_RUN=$1

if [ -n "$CAM_TO_RUN" ]; then
    # Run only specified camera
    echo "Running script for fetching from camera no $CAM_TO_RUN"
    python $REPO_DIR/fetch_save_stream.py -cl $REPO_DIR/custom_cam_info.json -l $RECORDINGS_DIR -cn "$CAM_TO_RUN" >> $OUTPUT_LOG 2>&1 &
else
    # Run all default cameras
    for cam in "${DEFAULT_CAMERAS[@]}"; do
        echo "Running script for fetching from camera no $cam"
        python $REPO_DIR/fetch_save_stream.py -cl $REPO_DIR/custom_cam_info.json -l $RECORDINGS_DIR -cn "$cam" >> $OUTPUT_LOG 2>&1 &
    done
fi

echo "fetch_save_stream scripts are running in the background!"
echo "Output is logged in location: $OUTPUT_LOG"
