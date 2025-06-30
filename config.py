import os

# Get the directory containing this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input/Output paths
INPUT_DIR = os.path.join(ROOT_DIR, 'input')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
STUBS_DIR = os.path.join(ROOT_DIR, 'stubs')

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, MODEL_DIR, STUBS_DIR]:
    os.makedirs(directory, exist_ok=True)

# File paths
INPUT_VIDEO = os.path.join(INPUT_DIR, '15sec_input_720p.mp4')
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, 'output.mp4')
MODEL_PATH = os.path.join(MODEL_DIR, 'best.pt')
STUB_PATH = os.path.join(STUBS_DIR, 'track_stubs.pkl') 