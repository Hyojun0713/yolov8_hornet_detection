import cv2
import os
from datetime import datetime
from google.cloud import storage


def draw_bounding_boxes(frame, detections):
    for detection in detections:
        box, conf, cls = detection
        x_center, y_center, w, h = box
        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 클래스와 신뢰도 표시
        label = f'{cls}: {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def save_image_locally(frame, track_id, prefix='hornet'):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"{prefix}_{track_id}_{current_time}.jpg"
    save_dir = r"C:\Users\SKTPJ12\Desktop\upload_detected_image"
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, image_filename)
    cv2.imwrite(image_path, frame)
    return image_path


def upload_to_gcs(image_path, bucket_name, gcs_filename):
    try:
        service_account_path = r"C:\Users\SKTPJ12\Documents\카카오톡 받은 파일\kulbul-dbd549773b0a.json"
        storage_client = storage.Client.from_service_account_json(service_account_path)

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_filename)
        blob.upload_from_filename(image_path)
        print(f"이미지가 GCS에 업로드되었습니다: gs://{bucket_name}/{gcs_filename}")
    except Exception as e:
        print(f"이미지 업로드 실패: {e}")

