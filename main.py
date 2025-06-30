from utils import read_video, save_video
from tracker import Tracker
import cv2
from colorassignment.assignment import TeamAssigner
import pickle
from config import INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH, STUB_PATH


def main():
    video_frames = read_video(INPUT_VIDEO)
    tracker = Tracker(MODEL_PATH)

    # First run will generate stubs, subsequent runs will use them
    tracks = tracker.track_player(video_frames, read_from_stub=False, stub_path=STUB_PATH)

    # Find a good frame for team color assignment 
    good_frame_idx = None
    min_players_needed = 4  # Need at least 4 players per team for good clustering
    
    for frame_num, players in enumerate(tracks['players']):
        if len(players) >= min_players_needed:
            good_frame_idx = frame_num
            break
    
    if good_frame_idx is None:
        raise ValueError("Could not find a frame with enough players for team assignment")
    
    color_assignment = TeamAssigner()
    # Assign team colors using the frame with enough players
    color_assignment.assign_team_color(video_frames[good_frame_idx], 
                                     tracks['players'][good_frame_idx])
    
    # Assign teams to all players across frames
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = color_assignment.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = color_assignment.team_colors[team]

    # Save the results
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, OUTPUT_VIDEO)

    # Now that we have good results, save them to stubs for future use
    with open(STUB_PATH, 'wb') as f:
        pickle.dump(tracks, f)


if __name__ == '__main__':
    main()




