#!/bin/bash

echo "Step 1: Creating virtual environment..."
python -m venv venv

echo "Activating virtual environment..."
source venv/Scripts/activate

echo "Step 2: Installing all requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Everything is ready."