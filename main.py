import os
from datetime import datetime
import config
from video_processor import process_video
from train import train_with_custom_data

def main():
    input_choice = input("video? or webcam? ").strip().lower()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(config.OUTPUT_DIR, f'output_{current_time}.mp4')

    if input_choice == "video":
        video_path = config.VIDEO_PATH  # config.py에서 비디오 파일 경로 사용
        process_video(video_path, output_filename)
    elif input_choice == "webcam":
        webcam_url = "http://172.23.248.239:8081"  # 웹캠 스트림 URL
        process_video(webcam_url, output_filename)
    else:
        print("잘못된 입력입니다. 'video' 또는 'webcam'을 입력하세요.")

if __name__ == '__main__':
    # final_model_path = train_with_custom_data()
    # print(f'Final model saved at: {final_model_path}')
    main()
