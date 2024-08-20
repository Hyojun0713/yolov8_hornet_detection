import requests
import config
from datetime import datetime

def send_to_backend(num_hornets):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    data = {
        "SN": config.DEVICE_SN,
        "num": num_hornets,
        "nt": datetime.now().strftime("%Y/%m/%d/%H")
    }
    try:
        response = requests.patch(config.BACKEND_URL, headers=headers, json=data)
        response.raise_for_status()

        print(f"서버 응답: {response.text}")  # 서버 응답을 텍스트 형태로 출력

        try:
            response_data = response.json()
            pastmax = response_data.get("pastmax", 3)  # "pastmax" 값이 없으면 기본값 3 사용
        except ValueError:
            print("응답을 JSON으로 파싱할 수 없습니다. 기본값을 사용합니다.")
            pastmax = 3

        print(f"데이터 전송 성공: {num_hornets}마리의 말벌 감지, pastmax: {pastmax}")
        return pastmax  # pastmax 값을 반환
    except requests.exceptions.RequestException as e:
        print(f"데이터 전송 실패: {e}")
        return 3  # 예외 발생 시 기본값 3 반환
