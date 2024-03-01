from inference import *

video_path = "/path/video.mp4"
json_pose = "/path/video_pose.json"
json_pose_tracking = "path/json_pose_tracking.json"

if __name__ == "__main__":
    
    test = VideoInference(video_path=video_path)
    test.add_model(['bytetrack_sort'])
    test.read_from_json(json_pose)
    test.process_video(output_json=json_pose_tracking)

