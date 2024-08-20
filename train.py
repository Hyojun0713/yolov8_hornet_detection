import os
import yaml
from ultralytics import YOLO
import config

def get_next_run_number(project_dir):
    existing_runs = [d for d in os.listdir(project_dir) if d.startswith('run_') and d[4:].isdigit()]
    if not existing_runs:
        return 1
    return max([int(d[4:]) for d in existing_runs]) + 1

def train_model(model, data_yaml, project_dir, epochs=config.EPOCHS):
    run_number = get_next_run_number(project_dir)
    run_name = f"run_{run_number}"

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=config.YOLO_IMG_SIZE,
        batch=config.YOLO_BATCH_SIZE,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.0001,
        warmup_epochs=3,
        patience=20,
        project=project_dir,
        name=run_name,
        augment=True,
        cos_lr=True,
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
    )

    save_dir = os.path.join(project_dir, run_name)
    model_save_path = os.path.join(save_dir, f'{run_name}_final.pt')
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")

    return model, model_save_path

def train_with_custom_data():
    # data.yaml 파일 경로 설정
    data_yaml_path = r'C:\Users\SKTPJ12\PycharmProjects\Test\data.yaml'

    # YOLO 모델 초기화
    model = YOLO(r'C:\Users\SKTPJ12\PycharmProjects\Test\Orginal_model\yolov8n.pt')  # 기존 모델을 불러옴
    model.model.nc = config.NUM_CLASSES  # 클래스 수 설정
    model.model.names = config.CLASS_NAMES  # 클래스 이름 설정

    # custom data (v2i 및 v7i 포함)로 학습
    model, final_model_path = train_model(model, data_yaml_path, config.PROJECT_DIR)

    return final_model_path
