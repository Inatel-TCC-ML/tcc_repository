import cv2
import os


def extract_frames_interval(
    video_path,
    output_dir,
    interval_seconds=5
):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise ValueError("Não foi possível identificar o FPS do vídeo.")

    interval_frames = max(1, round(fps * interval_seconds))

    extracted_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % interval_frames == 0:
            timestamp = frame_idx / fps

            filename = (
                f"frame_{frame_idx:06d}_"
                f"{timestamp:.2f}s.jpg"
            )

            output_path = os.path.join(
                output_dir,
                filename
            )

            cv2.imwrite(output_path, frame)

            extracted_frames.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "path": output_path
            })

        frame_idx += 1

    cap.release()

    return extracted_frames