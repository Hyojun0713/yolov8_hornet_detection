import cv2
import time
import os
from detector import HornetDetector
from tracker import HornetTracker
from backend_communicator import send_to_backend
from image_utils import save_image_locally, upload_to_gcs, draw_bounding_boxes
import config
from datetime import datetime

def process_video(input_source, output_filename):
    detector = HornetDetector()
    tracker = HornetTracker()

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"{input_source}를 열 수 없습니다.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    detection_start_time = None
    best_frame = None
    max_hornets_detected = 0
    last_detection_time = time.time()  # 마지막 감지 시간을 현재 시간으로 초기화
    pastmax = 3  # 초기값 설정

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다. 스트림이 중지되었거나 연결이 끊어졌습니다.")
            break

        detections = detector.detect(frame)
        num_hornets = len(detections)
        tracks = tracker.update(detections, frame)

        if num_hornets > 0:
            last_detection_time = time.time()  # 말벌이 감지되면 마지막 감지 시간 갱신
            if detection_start_time is None:
                detection_start_time = time.time()
                print("말벌 감지 시작!")

            if num_hornets > max_hornets_detected:
                max_hornets_detected = num_hornets
                best_frame = frame.copy()

            # pastmax보다 감지된 말벌 수가 크다면 업로드
            if max_hornets_detected > pastmax:
                if best_frame is not None:
                    frame_with_boxes = draw_bounding_boxes(best_frame, detections)
                    image_path = save_image_locally(frame_with_boxes, config.DEVICE_SN)
                    gcs_filename = os.path.join(f"SN_{config.DEVICE_SN}", datetime.now().strftime("%Y-%m-%d-%H.jpg"))
                    upload_to_gcs(image_path, 'kulbul', gcs_filename)
                detection_start_time = None
                max_hornets_detected = 0
                best_frame = None


        else:
            # 5분 동안 말벌이 감지되지 않으면 종료
            if time.time() - last_detection_time >= 300:
                print("5분 동안 말벌이 감지되지 않아 감지를 종료합니다.")
                break

        for track in tracks:
            if track.track_id in tracker.current_hornets:
                ltrb = track.to_ltrb()
                cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'Hornet ID: {track.track_id}', (int(ltrb[0]), int(ltrb[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f'Current Hornets: {len(tracker.current_hornets)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 백엔드로 데이터 전송 및 pastmax 값 갱신
        new_pastmax = send_to_backend(len(tracker.current_hornets))
        if new_pastmax is not None:
            pastmax = new_pastmax  # 응답이 있으면 pastmax 값을 갱신

        # pastmax보다 감지된 말벌 수가 크다면 업로드
        if len(tracker.current_hornets) > pastmax:
            if best_frame is not None:
                frame_with_boxes = draw_bounding_boxes(best_frame, detections)
                image_path = save_image_locally(frame_with_boxes, config.DEVICE_SN)
                gcs_filename = os.path.join(f"SN_{config.DEVICE_SN}", datetime.now().strftime("%Y-%m-%d-%H.jpg"))
                upload_to_gcs(image_path, 'kulbul', gcs_filename)

        out.write(frame)

    cap.release()
    out.release()
    print(f"처리된 영상이 '{output_filename}'에 저장되었습니다.")
