from flask import Flask, request
import numpy as np
import cv2
import boto3
from ultralytics import YOLO
from flask_cors import CORS
from video_handler import *

# AWS 자격 증명 설정
AWS_ACCESS_KEY_ID = 'your-access-key-id'
AWS_SECRET_ACCESS_KEY = 'your-secret-access-key'
AWS_REGION = 'your-region'  # 예: 'us-east-1'

# AWS S3 클라이언트 생성
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# 버킷 이름과 영상 파일 이름 설정
BUCKET_NAME = 'your-bucket-name'
VIDEO_FILE_KEY = 'path/to/your/video.mp4'
app = Flask(__name__)
CORS(app)

@app.route("/yolo1", methods=["GET"])
def health_check():
    return "yeah~"

# 이미지를 업로드하는 엔드포인트
@app.route("/yolo1/upload", methods=["POST"])
def upload():
    try:
        # Step 1:
        # mp4 영상 s3에서 끌어오기

        # Step 2:
        # 가져온 영상 프레임 별로 분할
        video = cv2.VideoCapture("resources/Short4Mosaicing.mp4")

        video_handler = VideoHandler(video)
        video_handler.run_detectors()

        # Step 3:
        # 프레임 별로 분할한 이미지 모델 사용하여 예측

        # Step 4:
        # 인식된 객체 포인트 json 파일로 저장.

        # 탐지된 객체를 반환
        print('123')
        return '123'
    except Exception as e:
        print(f"요청 처리 중 오류 발생: {str(e)}")
        return "오류", 400

if __name__ == "__main__":
    app.run()
