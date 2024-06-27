from flask import Flask, jsonify, Response, request, send_file
import numpy as np
import cv2
import boto3
import shutil
from ultralytics import YOLO
from flask_cors import CORS
from video_handler import *
from botocore.exceptions import NoCredentialsError

# AWS S3 클라이언트 생성
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# 버킷 이름과 영상 파일 이름 설정
BUCKET_NAME = 'gasby-req'
app = Flask(__name__)
CORS(app)

@app.route("/yolo1", methods=["GET"])
def health_check():
    return "yeah~"

# 이미지를 업로드하는 엔드포인트
# @app.route("/yolo1/upload", methods=["POST"])
def yolo_detection(source):
    try:
        # Step 1:
        # mp4 영상 s3에서 끌어오기

        # Step 2:
        # 가져온 영상 프레임 별로 분할
        source = './resources/Short4Mosaicing.mp4'
        video = cv2.VideoCapture(source)

        video_handler = VideoHandler(video)
        video_handler.run_detectors()

        # Step 3:
        # 프레임 별로 분할한 이미지 모델 사용하여 예측

        # Step 4:
        # 인식된 객체 포인트 json 파일로 저장.

        # 탐지된 객체를 반환
        return 'detection success'
    except Exception as e:
        print(f"요청 처리 중 오류 발생: {str(e)}")
        return "오류", 400


@app.route('/video', methods=['POST'])
def get_video():
    data = request.get_json()
    payload = data.get('payload') if data else None
    try:
        # 폴더 내 모든 파일 가져오기
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=payload)
        
        file_name = ''
        if 'Contents' in response:
            files = [content['Key'] for content in response['Contents']]
            file_name = files[0]
            print(file_name)
        else:
            return jsonify({'files': []})
        
        LOCAL_FILE_PATH = './video/' + file_name
        local_dir = os.path.dirname(LOCAL_FILE_PATH)
        os.makedirs(local_dir, exist_ok=True)
        # S3에서 영상 파일 가져와 로컬 파일로 저장
        with open(LOCAL_FILE_PATH, 'wb') as f:
            s3.download_fileobj(BUCKET_NAME, file_name, f)
        
        yolo_detection(LOCAL_FILE_PATH)

        shutil.rmtree(local_dir)

        # 저장된 파일을 클라이언트에게 제공
        return '123'
    except NoCredentialsError:
        return "AWS 자격 증명이 설정되지 않았습니다.", 403
    except s3.exceptions.NoSuchKey:
        return "해당 영상 파일을 찾을 수 없습니다.", 404
    except Exception as e:
        return str(e), 500
    

if __name__ == "__main__":
    app.run()
