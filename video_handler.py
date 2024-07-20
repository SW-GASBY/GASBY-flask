# Divide the video to mp4 as frame & Detecting Object by YOLO model
import os
import cv2
import re
import numpy as np
from ultralytics import YOLO
import json
import pickle
from player_tracking import player_tracking
from shapely.geometry import Point, Polygon

# 학습된 모델 파일의 경로
detection_model_path = "resources/weights/detection/best.pt"
segmentation_model_path = "resources/weights/segmentation/best.pt"
classify_model_path = "resources/weights/classify/best.pt"

# 학습된 모델을 로드
detection_model = YOLO(detection_model_path)
segmentation_model = YOLO(segmentation_model_path)
classify_model = YOLO(classify_model_path)

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

        x1, y1 = int(box['x1']), int(box['y1'])
        x2, y2 = int(box['x2']), int(box['y2'])

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if label == "player":
            confidence = detection['uniform_color']
            label_text = f"{label} ({confidence})"
        else: # 라벨 작성
            label_text = f"{label}"
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

# 유니폼 색상 분류 함수
def classify_uniform_color(image, model, teamA, teamB):
    results = model(image)
    for result in results:
        if hasattr(result, 'probs'):
            top5_list = result.probs.top5
            for i in range(5):
                if result.names[top5_list[i]] == teamA or result.names[top5_list[i]] == teamB:
                    return result.names[top5_list[i]]
            # top_class = result.names[result.probs.top1]
            # return top_class
    return None

TOPCUT = 320

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

list = []
class VideoHandler:
    def __init__(self, video, frame_skip=3):
        self.video = video
        self.frames = []
        self.frame_skip = frame_skip # frame_skip이 1이면 모든 프레임 처리, 2이면 매 2번째 프레임만 처리

    def run_detectors(self, source, teamA, teamB):
        i = 0
        frame_count = 0
        while self.video.isOpened():
            ok, frame = self.video.read()

            if not ok:
                break
            else:
                if frame_count % self.frame_skip == 0:
                    # 프레임을 리스트에 추가
                    self.frames.append(frame)
                    
                    # 이미지를 객체로 탐지
                    detections = detect_objects(frame, detection_model)
                    
                    # 탐지 결과 출력
                    # print(f"Frame {frame_count} detections: {detections}")
                    
                    if len(detections) != 0:
                        # 세그멘테이션 결과를 detection 결과와 사용
                        segmentations = detect_objects(frame, segmentation_model)
                        detections = check_detection_in_segmentation(detections, segmentations)
                        for detection in detections:
                            box = detection['box']
                            label = detection['name']
                            x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])
                            cropped_image = frame[y1:y2, x1:x2]

                            if label == 'player':
                                uniform_color = classify_uniform_color(cropped_image, classify_model, teamA, teamB)
                                detection["uniform_color"] = uniform_color
                                
                        list.append(detections)
                        
                    # 탐지 결과에 따라 바운딩 박스 그리고 라벨 작성
                    image_with_boxes = draw_boxes(frame, detections)
                    os.makedirs(source + '/image', exist_ok=True)

                    # 이미지를 로컬에 저장
                    output_path = source + '/image/output_image'+str(i)+'.jpg'
                    i += 1
                    cv2.imwrite(output_path, image_with_boxes)
                    # print(f"이미지 저장 완료: {output_path}")
                
                frame_count += 1

        # JSON 파일 저장 경로 수정
        save_list_to_json(list, source + '/data.json')
        
        # 모든 프레임을 pkl 파일로 저장
        with open(source + '/frames.pkl', 'wb') as f:
            pickle.dump(self.frames, f)

        player_tracking(source, teamA, teamB)


# JSON 파일로 저장하는 함수
def save_list_to_json(list_data, file_name):
    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(list_data, json_file, ensure_ascii=False, indent=4)


def check_detection_in_segmentation(detections, segmentations):
    new_detection = []
    for detection in detections:
        if detection['name'] != 'basketball':
            # 바운딩 박스의 중심 좌표 계산
            box = detection['box']
            center_x = (box['x1'] + box['x2']) / 2
            center_y = box['y2']
            center_point = Point(center_x, center_y)
            
            isInside = False
            for segmentation in segmentations:
                # 세그멘테이션 폴리곤 생성
                polygon_points = [(x, y) for x, y in zip(segmentation['segments']['x'], segmentation['segments']['y'])]
                polygon = Polygon(polygon_points)
                
                # 바운딩 박스 중심이 세그멘테이션 폴리곤 내에 있는지 확인
                if polygon.contains(center_point):
                    if segmentation['name'] == 'basketball-court':
                        if 'position_name' not in detection:
                            detection['position_name'] = segmentation['name']        
                    else:
                        detection['position_name'] = segmentation['name']
                    isInside = True
            if isInside == True:
                new_detection.append(detection)
        else:
            new_detection.append(detection)

            
    return new_detection