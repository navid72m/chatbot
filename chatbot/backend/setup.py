# setup.py - Two-stage approach
from setuptools import setup

APP = ['wrapper.py']  # A simple wrapper script that launches your actual app
DATA_FILES = [('', ['main.py', 'vector_store.py'])]  # Include your actual scripts as data files
OPTIONS = {
    'argv_emulation': False,
    'packages': ['fastapi', 'uvicorn'],
    'includes': ['numpy'],
    'semi_standalone': True,  # This can help with complex dependencies
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)