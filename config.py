import os

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 비디오 및 모델 설정
VIDEO_PATH = r"C:\Users\SKTPJ12\Downloads\My Pet Wasp Colony Escaped….mp4"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'hornet_detection/letsgobee4.pt')

# 검출 및 추적 설정
CONFIDENCE_THRESHOLD = 0.5
MIN_HITS = 3
DETECTION_PERSISTENCE = 0.3
HORNET_THRESHOLD = 3

# 백엔드 통신 설정
BACKEND_URL = "http://kulbul.iptime.org:8000/detector/vision"
DEVICE_SN = "111111111111"

# 학습 관련 설정
TRAIN_IMG_DIRS = [os.path.join(DATA_DIR, 'train', 'images')]
VAL_IMG_DIRS = [os.path.join(DATA_DIR, 'valid', 'images')]
TEST_IMG_DIRS = [os.path.join(DATA_DIR, 'test', 'images')]
SMALL_IMAGE_AUGMENTATION = 5
BATCH_SIZE = 16
EPOCHS = 100
PROJECT_DIR = os.path.join(BASE_DIR, "hornet_detection")

# 데이터셋 관련 설정
NUM_CLASSES = 2
CLASS_NAMES = ['bee', 'hornet']

# 출력 설정
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO 모델 설정
YOLO_IMG_SIZE = 640
YOLO_BATCH_SIZE = 16

# 학습 설정
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.937

# 데이터 증강 설정
MOSAIC = 0.5
MIXUP = 0.2

# 하드웨어 설정
NUM_WORKERS = 4