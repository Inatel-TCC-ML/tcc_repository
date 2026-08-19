from extract_keyframe import extract_keyframes_ssim

video_path = "../../datasets/kagglehub/datasets/kanzeus/realai-video-dataset/versions/1/ai/ai (1).mp4"

extract_keyframes_ssim(video_path, output_dir="output", threshold=0.45, resize_dim=(320, 180))
