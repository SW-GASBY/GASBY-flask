# Main Server API File.

from flask import Flask, jsonify, request
import cv2
import boto3
import shutil
from flask_cors import CORS
from video_handler import *
from dotenv import dotenv_values
from botocore.exceptions import NoCredentialsError


env = dotenv_values('.env')
AWS_ACCESS_KEY = env['AWS_ACCESS_KEY_ID']
AWS_SECRET_KEY = env['AWS_SECRET_ACCESS_KEY']
AWS_REGION = env['AWS_REGION']

# AWS S3 클라이언트 생성
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)


# # AWS S3 클라이언트 생성
# s3 = boto3.client(
#     's3',
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#     region_name=os.getenv('AWS_REGION')
# )

# 버킷 이름과 영상 파일 이름 설정
BUCKET_NAME = 'gasby-req'
app = Flask(__name__)
CORS(app)

@app.route("/yolo-predict", methods=["GET"])
def health_check():
    return "yeah~"

@app.route("/yolo-predict/upload", methods=["POST"])
def get_video():
    # 요청 수신 시간을 기록합니다.
    start_time = time.time()

    data = request.get_json()
    payload = data.get('payload') if data else None
    teamA = data.get('team_a_color') if data else None
    teamB = data.get('team_b_color') if data else None
    
    try:
        # Step 1:
        # mp4 영상 s3에서 끌어오기

        # 폴더 내 모든 파일 가져오기
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=payload)
        
        file_name = ''
        if 'Contents' in response:
            files = [content['Key'] for content in response['Contents']]
            file_name = files[1]
            json_file = files[0]
            print(file_name)
        else:
            return jsonify({'files': []})
        
        LOCAL_FILE_PATH = './video/' + file_name
        local_dir = os.path.dirname(LOCAL_FILE_PATH)
        os.makedirs(local_dir, exist_ok=True)
        # S3에서 영상 파일 가져와 로컬 파일로 저장
        with open(LOCAL_FILE_PATH, 'wb') as f:
            s3.download_fileobj(BUCKET_NAME, file_name, f)
        with open('./video/' + json_file, 'wb') as f:
            s3.download_fileobj(BUCKET_NAME, json_file, f)

        # 파일을 열고 내용을 읽어옵니다
        with open('./video/' + json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        teamA = data['team_a_color']
        teamB = data['team_b_color']

        # Step 2: & Step 3: & Step 4:
        # 가져온 영상 프레임 별로 분할
        # 프레임 별로 분할한 이미지 모델 사용하여 예측
        # 인식된 객체 포인트 json 파일로 저장.
        yolo_detection(LOCAL_FILE_PATH, teamA, teamB)
        
        # player_tracking(os.path.dirname(LOCAL_FILE_PATH))

        # 저장된 파일을 클라이언트에게 제공
        # 저장된 파일 s3로 업로드
        file_name1 = 'video/' + payload + '/'
        bucket_name1 = 'gasby-mot-result'
        object_name = payload + '/'  # Optional
        # s3.upload_file(file_name1 + 'frames.pkl', bucket_name1, object_name + 'frames.pkl')
        s3.upload_file(file_name1 + 'ball.json', bucket_name1, object_name + payload + '_ball.json')
        s3.upload_file(file_name1 + 'player_positions_filtered.json', bucket_name1, object_name + payload + '.json')
        # 로컬파일경로 + 파일명 + 파일종류, 버킷명, s3버킷의 원하는경로 + 파일명 + 파일종류

        ################################################################ 밑에 주석 풀면 로컬파일 삭제 됨. ################################################################
        shutil.rmtree(local_dir)
        # 예제 응답 데이터
        response_data = {"message": "Received"}
        
        # 요청 처리 종료 시간을 기록합니다.
        end_time = time.time()
        
        # 걸린 시간을 계산합니다.
        elapsed_time = end_time - start_time
        
        # 응답 데이터에 소요 시간을 추가합니다.
        response_data["elapsed_time"] = elapsed_time
        
        return jsonify(response_data)
        return '123'
    except NoCredentialsError:
        return "AWS 자격 증명이 설정되지 않았습니다.", 403
    except s3.exceptions.NoSuchKey:
        return "해당 영상 파일을 찾을 수 없습니다.", 404
    except Exception as e:
        return str(e), 500

# yolo 호출하는 함수
def yolo_detection(source, A, B):
    try:
        # source = './resources/Short4Mosaicing.mp4'
        video = cv2.VideoCapture(source)

        video_handler = VideoHandler(video)
        video_handler.run_detectors(os.path.dirname(source), A, B)
    except Exception as e:
        print(f"요청 처리 중 오류 발생: {str(e)}")
        return "오류", 400

import time

@app.route('/api/time', methods=['POST'])
def api_time():
    # 요청 수신 시간을 기록합니다.
    start_time = time.time()
    
    # 실제로 처리할 로직을 여기에 추가합니다.
    data = request.json
    # 예제 응답 데이터
    response_data = {"message": "Received", "data": data}
    
    # 요청 처리 종료 시간을 기록합니다.
    end_time = time.time()
    
    # 걸린 시간을 계산합니다.
    elapsed_time = end_time - start_time
    
    # 응답 데이터에 소요 시간을 추가합니다.
    response_data["elapsed_time"] = elapsed_time
    
    return jsonify(response_data)

if __name__ == "__main__":
    app.run()
