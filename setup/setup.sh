#!/bin/bash

echo "Setting up the Python virtual environment..."
python3 -m venv venv

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r setup/requirements.txt

echo "Setup complete! Run 'source venv/bin/activate' to start."