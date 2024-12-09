#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR=$(dirname "$SCRIPT_DIR")
# echo "The repo location is: $REPO_DIR"

# echo "Creating virtual environment"
source $REPO_DIR/home_cam/bin/activate

# echo "Running script for managing directory and DB"
OUTPUT_LOG=$REPO_DIR/results/telegram_bot.txt
LOG_DIR=$REPO_DIR/telegram_bot_log
python $REPO_DIR/telegram_bot.py -cl $1 -fl $2 -msg $3 -l $LOG_DIR >> $OUTPUT_LOG 2>&1 &
echo "telegram bot is running in the background!"
# echo "Output is logged in location: $OUTPUT_LOG"