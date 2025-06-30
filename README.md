# Player Identification System

A system for tracking and identifying players in football videos using YOLO and ByteTrack.

## Directory Structure

```
playeridentification/
├── input/                  # Input video files
│   └── 15sec_input_720p.mp4
├── output/                 # Output video files
├── model/                  # YOLO model files
│   └── best.pt
├── stubs/                  # Cached tracking results
├── tracker/               # Tracking implementation
├── colorassignment/      # Team color assignment
└── utils/                # Utility functions
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your input video in the `input/` directory as `15sec_input_720p.mp4`

3. Place your YOLO model in the `model/` directory as `best.pt`

4. Run the system:
```bash
python main.py
```

The output video will be saved in the `output/` directory.

## Features

- Player detection using YOLO
- Player tracking using ByteTrack
- Team assignment using color clustering (k means)
- Caching of tracking results for faster subsequent runs 