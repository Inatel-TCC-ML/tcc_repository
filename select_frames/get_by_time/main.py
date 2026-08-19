from extract_frame import extract_frames_interval

video_path = "../../datasets/kagglehub/datasets/kanzeus/realai-video-dataset/versions/1/ai/ai (1).mp4"

extract_frames_interval(
    video_path,
    output_dir="output",
    interval_seconds=5
)