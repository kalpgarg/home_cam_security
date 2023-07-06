#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR=$(dirname "$SCRIPT_DIR")
echo "The repo location is: $REPO_DIR"

echo "Creating virtual environment"
source $REPO_DIR/home_cam/bin/activate

OUTPUT_LOG=$REPO_DIR/results/script_mon.txt
python $REPO_DIR/script_mon.py >> $OUTPUT_LOG 2>&1 &

echo "script monitor is running in the background!"
echo "Output is logged in location: $OUTPUT_LOG"