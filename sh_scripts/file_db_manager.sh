#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR=$(dirname "$SCRIPT_DIR")
echo "The repo location is: $REPO_DIR"

echo "Creating virtual environment"
source $REPO_DIR/home_cam/bin/activate

echo "Running script for managing directory and DB"
OUTPUT_LOG=$REPO_DIR/results/file_manager.txt
LOG_DIR=$REPO_DIR/cam_stream_log
RECORDINGS_DIR=$REPO_DIR/cam_stream_log/recordings
DB_PATH=$REPO_DIR/instance/user_db.db
python $REPO_DIR/file_manager.py -cl $REPO_DIR/custom_cam_info.json -db $DB_PATH -if $RECORDINGS_DIR -l $LOG_DIR -cn 1 >> $OUTPUT_LOG 2>&1 &
python $REPO_DIR/file_manager.py -cl $REPO_DIR/custom_cam_info.json -db $DB_PATH -if $RECORDINGS_DIR -l $LOG_DIR -cn 2 >> $OUTPUT_LOG 2>&1 &
python $REPO_DIR/file_manager.py -cl $REPO_DIR/custom_cam_info.json -db $DB_PATH -if $RECORDINGS_DIR -l $LOG_DIR -cn 3 >> $OUTPUT_LOG 2>&1 &
python $REPO_DIR/file_manager.py -cl $REPO_DIR/custom_cam_info.json -db $DB_PATH -if $RECORDINGS_DIR -l $LOG_DIR -cn 4 >> $OUTPUT_LOG 2>&1 &
echo "File manager is running in the background!"
echo "Output is logged in location: $OUTPUT_LOG"

