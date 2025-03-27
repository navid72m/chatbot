#!/usr/bin/env python3
import os
import sys
import subprocess

print("🚀 Starting wrapper...")

# Get the directory containing this script
app_dir = os.path.dirname(os.path.abspath(__file__))
print(f"📁 App directory: {app_dir}")

# Set environment variable
os.environ['PYTHONPATH'] = app_dir
print(f"🐍 Using Python executable: {sys.executable}")

# Build the path to main.py
main_script = os.path.join(app_dir, 'main.py')
print(f"📄 Running main.py at: {main_script}")

# Check if main.py exists
if not os.path.exists(main_script):
    print("❌ main.py not found!")
    sys.exit(1)

# Launch main.py
result = subprocess.run(['/Users/seyednavidmirnourilangeroudi/miniconda3/bin/python', os.path.join(app_dir, 'main.py')])

print(f"✅ Subprocess exited with code: {result.returncode}")
