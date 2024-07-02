# Divide the video to mp4 as frame & Detecting Object by YOLO model
import os
import cv2
import re
import numpy as np
from ultralytics import YOLO
import json
from player_tracking import player_tracking

# 학습된 모델 파일의 경로
best_model_path = "yolov8x.pt"

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

def numerical_sort(value):
    """ 숫자를 기준으로 정렬하기 위한 함수 """
    parts = re.split(r'(\d+)', value)
    parts[1::2] = map(int, parts[1::2])
    return parts

image_folder = './image'  # 이미지가 저장된 폴더 경로
def images_to_video(image_folder, output_video_path, frame_rate):
    # 이미지 파일들을 프레임 순서대로 정렬
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort(key=numerical_sort)

    # 첫 번째 이미지로 비디오의 높이와 너비를 설정
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # 비디오 라이터 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (MP4)
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # 각 이미지를 비디오에 추가
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 비디오 라이터 객체 해제
    video.release()

# csv_file_path = "label.csv"

TOPCUT = 320

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

list = []
class VideoHandler:
    def __init__(self, video):
        self.video = video
        self.frames = []
        self.d_frames = []

    def run_detectors(self, source):
        i = 0
        while self.video.isOpened():
            ok, frame = self.video.read()

            if not ok:
                break
            else:
                # 이미지를 객체로 탐지
                detections = detect_objects(frame, best_model)

                # print(detections)
                if len(detections) != 0:
                    list.append(detections)

                # 탐지 결과에 따라 바운딩 박스 그리고 라벨 작성
                image_with_boxes = draw_boxes(frame, detections)
                os.makedirs(source + '/image', exist_ok=True)

                # 이미지를 로컬에 저장
                output_path = source + '/image/output_image'+str(i)+'.jpg'
                i += 1
                cv2.imwrite(output_path, image_with_boxes)
                print(f"이미지 저장 완료: {output_path}")

        # # 예시 사용법
        # output_video_path = 'demo1.mp4'  # 출력 비디오 파일 경로
        # frame_rate = 60  # 프레임 레이트 (FPS)

        # images_to_video(image_folder, output_video_path, frame_rate)

        # 함수 호출 예시
        # print(source)
        save_list_to_json(list, source + '/data.json')
        player_tracking(source)


# JSON 파일로 저장하는 함수
def save_list_to_json(list_data, file_name):
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(list_data, json_file, ensure_ascii=False, indent=4)
