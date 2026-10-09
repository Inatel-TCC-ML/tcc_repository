import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_IMAGE_DIR = os.path.join(ROOT_DIR, "..", "datasets", "input")
INPUT_IMAGE_DIR = os.path.normpath(INPUT_IMAGE_DIR)

# ---------- CNNDetection ---------- #
CNNDETECTION_DIR = os.path.join(ROOT_DIR, "..", "external", "CNNDetection")
CNNDETECTION_DIR = os.path.normpath(CNNDETECTION_DIR)

WEIGHTS_PATH = os.path.join(ROOT_DIR, "..", "weights", "blur_jpg_prob0.5.pth")
WEIGHTS_PATH = os.path.normpath(WEIGHTS_PATH)

# ---------- UniversalFakeDetect ---------- #
UNIVFD_DIR = os.path.join(ROOT_DIR, "..", "external", "UniversalFakeDetect")
UNIVFD_DIR = os.path.normpath(UNIVFD_DIR)

UNIVFD_WEIGHTS_PATH = os.path.join(UNIVFD_DIR, "pretrained_weights", "fc_weights.pth")
UNIVFD_WEIGHTS_PATH = os.path.normpath(UNIVFD_WEIGHTS_PATH)

# ---------- NPRDeepfakeDetect ---------- #
NPRDEEPFAKE_DIR = os.path.join(ROOT_DIR, "..", "external", "NPR-DeepfakeDetection")
NPRDEEPFAKE_DIR = os.path.normpath(NPRDEEPFAKE_DIR)

NPRDEEPFAKE_WEIGHTS_PATH = os.path.join(NPRDEEPFAKE_DIR, "NPR.pth")
NPRDEEPFAKE_WEIGHTS_PATH = os.path.normpath(NPRDEEPFAKE_WEIGHTS_PATH)

# ---------- B-Free ---------- #
BFREE_DIR = os.path.join(ROOT_DIR, "..", "external", "B-Free", "code")
BFREE_DIR = os.path.normpath(BFREE_DIR)

BFREE_WEIGHTS_DIR = os.path.join(ROOT_DIR, "..", "weights", "BFREE_dino2reg4")
BFREE_WEIGHTS_DIR = os.path.normpath(BFREE_WEIGHTS_DIR)
