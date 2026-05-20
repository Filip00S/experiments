#!/bin/bash

# Path to your experiments directory
EXPER_DIR="/storage/YOUR_SERVER/home/username/experiments"

echo "Setting executable permissions for all .sh scripts..."
find "$EXPER_DIR" -type f -name "*.sh" -exec chmod u+x {} \;

echo "Ensuring the datasets directory exists and has proper read/write permissions..."
mkdir -p "$EXPER_DIR/datasets"
chmod -R u+rwX "$EXPER_DIR/datasets"

echo "Ensuring the logs directory exists and has proper read/write permissions..."
mkdir -p "$EXPER_DIR/scripts/logs"
chmod -R u+rwX "$EXPER_DIR/scripts/logs"

echo "Permissions successfully granted! You can now run your scripts normally."
