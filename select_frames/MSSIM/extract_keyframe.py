import cv2
from skimage.metrics import structural_similarity as ssim


def extract_keyframes_ssim(video_path, output_dir, threshold=0.85, resize_dim=(320, 180)):
    cap = cv2.VideoCapture(video_path)
    print(cap.isOpened())
    keyframes = []
    last_keyframe_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # reduz resolução só para o cálculo do SSIM
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, resize_dim)

        if last_keyframe_gray is None:
            # primeiro frame sempre é keyframe
            is_keyframe = True
        else:
            score, _ = ssim(last_keyframe_gray, gray_small, full=True)
            is_keyframe = float(score) < float(threshold)

        if is_keyframe:
            keyframes.append(frame_idx)
            last_keyframe_gray = gray_small
            cv2.imwrite(f"{output_dir}/frame_{frame_idx:06d}.jpg", frame)

        frame_idx += 1

    cap.release()
    return keyframes