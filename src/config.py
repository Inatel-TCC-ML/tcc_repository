import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CNNDETECTION_DIR = os.path.join(ROOT_DIR, "..", "external", "CNNDetection")
CNNDETECTION_DIR = os.path.normpath(CNNDETECTION_DIR)

WEIGHTS_PATH = os.path.join(ROOT_DIR, "..", "weights", "blur_jpg_prob0.5.pth")
WEIGHTS_PATH = os.path.normpath(WEIGHTS_PATH)

INPUT_IMAGE_DIR = os.path.join(ROOT_DIR, "..", "datasets", "input")
INPUT_IMAGE_DIR = os.path.normpath(INPUT_IMAGE_DIR)

INPUT_VIDEO_DIR = os.path.join(ROOT_DIR, "..", "datasets", "videos", "input")
INPUT_VIDEO_DIR = os.path.normpath(INPUT_VIDEO_DIR)

OUTPUT_DIR = os.path.join(ROOT_DIR, "..", "output")
OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)

UNIVFD_DIR = os.path.join(ROOT_DIR, "..", "external", "UniversalFakeDetect")
UNIVFD_DIR = os.path.normpath(UNIVFD_DIR)

UNIVFD_WEIGHTS_PATH = os.path.join(UNIVFD_DIR, "pretrained_weights", "fc_weights.pth")
UNIVFD_WEIGHTS_PATH = os.path.normpath(UNIVFD_WEIGHTS_PATH)
