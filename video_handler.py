import os
import cv2
import numpy as np
import skvideo.io
# import csv
from ultralytics import YOLO
import json
# 학습된 모델 파일의 경로
best_model_path = "yolov8m.pt"

# 학습된 모델을 로드
best_model = YOLO(best_model_path)

# 이미지를 탐지하는 함수
def detect_objects(image, model):
    # 이미지를 YOLO 모델에 입력
    results = model.predict(image)
    
    res_json = json.loads(results[0].tojson())

    return res_json

# 바운딩 박스를 그리고 라벨을 작성하는 함수
def draw_boxes(image, detections):
    for detection in detections:
        box = detection['box']
        label = detection['name']
        confidence = detection['confidence']

        x1, y1 = int(box['x1']), int(box['y1'])
        x2, y2 = int(box['x2']), int(box['y2'])

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 라벨 작성
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

# csv_file_path = "label.csv"

TOPCUT = 320

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


class VideoHandler:
    def __init__(self, video):
        self.video = video
        self.frames = []
        self.d_frames = []

    def run_detectors(self):
        i = 0
        while self.video.isOpened():
            ok, frame = self.video.read()

            if not ok:
                break
            else:
                # 이미지를 객체로 탐지
                detections = detect_objects(frame, best_model)

                # 탐지 결과에 따라 바운딩 박스 그리고 라벨 작성
                image_with_boxes = draw_boxes(frame, detections)

                # 이미지를 로컬에 저장
                output_path = './image/output_image'+str(i)+'.jpg'
                i += 1
                cv2.imwrite(output_path, image_with_boxes)
                print(f"이미지 저장 완료: {output_path}")
            