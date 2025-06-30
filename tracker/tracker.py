from ultralytics import YOLO
import supervision as sv
import sys
import cv2
import pickle
import os
import numpy as np
from collections import defaultdict
sys.path.append('../')
from utils import get_centre, get_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.6,  # threshold for track activation   
            lost_track_buffer=45,           # how many frames to keep dead tracks
            minimum_matching_threshold=0.85, # IoU threshold for matching
            frame_rate=30,                  # FPS of input video
            minimum_consecutive_frames=3     # no. of consec. frames before creating a track
        )
        self.track_history = defaultdict(list)  # Store trackk history
        self.appearance_features = {}  # Store appearance features per track
        
    def extract_features(self, frame, bbox):
        """Extract appearance features from player patch"""
        x1, y1, x2, y2 = [int(b) for b in bbox]
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        
        # Calc. color histogram 
        hist = cv2.calcHist([patch], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def update_track_history(self, track_id, bbox, frame_num):
        """Update track history with new position"""
        x_center, y_center = get_centre(bbox)
        self.track_history[track_id].append((frame_num, (x_center, y_center)))
        # Keeping only last 60 frames (as the video is @ 30fps, so that means 2 sec of history)
        if len(self.track_history[track_id]) > 60:
            self.track_history[track_id].pop(0)
    
    def predict_next_position(self, track_id):
        """Predict next position based on track history"""
        history = self.track_history[track_id]
        if len(history) < 2: 
            return None
        
        
        last_pos = history[-1][1]
        prev_pos = history[-2][1]
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        pred_x = last_pos[0] + dx
        pred_y = last_pos[1] + dy
        return (pred_x, pred_y)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.35)
            detections += detections_batch
        return detections

    def track_player(self, frames, read_from_stub=False, stub_path=None): 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        
        tracks = {
            'players':[],
            'referees':[],
            'ball':[]
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inverse = {v:k for k,v in cls_name.items()}
            
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Converting gk to player
            for obj_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_name[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_idx] = cls_name_inverse['player']

            # Track players with motion prediction
            track_player = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            # Process tracked objects
            for frame_detection in track_player:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Extract features and update history for players
                if cls_id == cls_name_inverse['player']:
                    # Update track history for motion prediction
                    self.update_track_history(track_id, bbox, frame_num)
                    
                    # Extract appearance features
                    features = self.extract_features(frames[frame_num], bbox)
                    if features is not None:
                        if track_id not in self.appearance_features:
                            self.appearance_features[track_id] = features
                        else:
                            # Update features with moving average
                            self.appearance_features[track_id] = 0.7 * self.appearance_features[track_id] + 0.3 * features
                    
                    # Predict next position
                    next_pos = self.predict_next_position(track_id)
                    
                    tracks['players'][frame_num][track_id] = {
                        "bbox": bbox,
                        "features": features.tolist() if features is not None else None,
                        "predicted_pos": next_pos
                    }

                if cls_id == cls_name_inverse['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox": bbox}
                
            # Handle ball tracking
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inverse['ball']:
                    tracks['ball'][frame_num] = {"bbox": bbox}

        if not read_from_stub and stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):
        x_centre, y_centre = get_centre(bbox)
        width = get_width(bbox)
        
        cv2.ellipse(
            frame,
            center=(int(x_centre), int(y_centre)),
            axes=(int(width//2), int(0.35*width//2)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_centre - rectangle_width//2
        x2_rect = x_centre + rectangle_width//2
        y1_rect = y_centre - rectangle_height//2 + 15
        y2_rect = y_centre + rectangle_height//2 + 15

        if track_id is not None:
            cv2.rectangle(frame,
                        (int(x1_rect), int(y1_rect)),
                        (int(x2_rect), int(y2_rect)),
                        color,
                        cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x, _ = get_centre(bbox)
        x = int(x)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ], dtype=np.int32)  # Ensure integer type
        
        # Reshape for OpenCV contour format
        triangle_points = triangle_points.reshape((-1,1,2))
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2)

        return frame
                
    def draw_annotations(self, frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            #player
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

            #referee
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'],(0,255,255), track_id)

            #ball   
            if ball_dict:
                frame = self.draw_triangle(frame, ball_dict['bbox'],(0,255,0))

            output_video_frames.append(frame)

        return output_video_frames