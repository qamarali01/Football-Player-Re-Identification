# Player Identification System: Technical Report

## Approach & Methodology

### Core Components
1. **Player Detection**: YOLO model for accurate player detection in video frames
2. **Player Tracking**: ByteTrack algorithm for maintaining consistent player IDs
3. **Team Assignment**: K-means clustering on player jersey colors
4. **Performance Optimization**: Caching system using stubs for faster subsequent runs

### Implementation Flow
1. Frame-by-frame video processing
2. Player detection and tracking with consistent IDs
3. Color-based team assignment using the first frame with at least 4 players
4. Result caching for improved performance

## Techniques Tried & Outcomes

### Tracking Improvements
1. **ByteTrack Parameters**:
   - Increased `lost_track_buffer` to 45 frames (1.5 seconds)
   - Set `track_activation_threshold` to 0.6
   - Adjusted `minimum_matching_threshold` to 0.85
   - Added `minimum_consecutive_frames=3` for stability

2. **Motion Prediction**:
   - Implemented track history (60 frames)
   - Added position prediction for occlusion handling
   - Improved tracking during player overlaps

3. **Appearance Features**:
   - Added color histogram features
   - Implemented moving average update for feature stability

### Team Assignment
1. **Color Clustering**:
   - Two-stage K-means clustering
   - First stage: Extract player colors from jersey regions
   - Second stage: Cluster players into teams
   - Removed hardcoded team assignments for better generalization

## Challenges Encountered

1. **ID Switching**:
   - Players getting new IDs after occlusions
   - Solved by tuning ByteTrack parameters and adding motion prediction

2. **Team Assignment**:
   - Initial hardcoded approach was brittle
   - Improved with dynamic color-based clustering

3. **Path Management**:
   - Initial hardcoded paths made the system non-portable
   - Solved by implementing centralized path configuration
   - Created proper directory structure for better organization

## Future Work

1. **Tracking Improvements**:
   - Implement appearance-based re-identification for better tracking after long occlusions
   - Add velocity prediction using Kalman filtering

2. **Team Assignment**:
   - Consider temporal consistency in team assignments:
     * Average player colors over multiple frames
     * Use voting mechanism for team decisions
     * Handle temporary color distortions
     * Maintain team consistency through occlusions

3. **Performance Optimization**:
   - Optimize batch size for detection (currently fixed at 20 frames)
   - Add GPU support for color processing

## Conclusion

The system successfully tracks players and assigns team identities. While the current implementation is functional, there are clear paths for improvement in tracking robustness, team assignment accuracy, and overall system performance. 