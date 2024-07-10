import os
import json
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# 학습된 모델 파일의 경로
best_model_path = "uniform.pt"

# 학습된 모델을 로드
best_model = YOLO(best_model_path)

# 이미지를 탐지하는 함수
def detect_objects(image, model):
    # 이미지를 YOLO 모델에 입력
    results = model(image)
    return results

def detect_objects_within_bounding_boxes(source_dir, model):
    # Load player positions from the JSON file
    with open(os.path.join(source_dir, 'data.json'), 'r') as file:
        player_positions = json.load(file)
    
    # Path to the directory containing the images
    image_dir = os.path.join(source_dir, 'image')
    output_dir = os.path.join(source_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_idx, frame_data in enumerate(player_positions):
        image_path = os.path.join(image_dir, f'output_image{frame_idx}.jpg')
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Image at {image_path} could not be loaded.")
            continue
        
        for obj_idx, obj_data in enumerate(frame_data):
            box = obj_data['box']
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            cropped_img = img[y1:y2, x1:x2]

            # Convert cropped image to PIL Image for YOLOv8
            cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            
            # Re-run YOLO model on the cropped image
            results = detect_objects(cropped_img_pil, model)
            
            # Save the detection results
            output_image_path = os.path.join(output_dir, f'frame_{frame_idx}_object_{obj_idx}.jpg')
            
            # Draw bounding boxes on the cropped image
            for result in results:
                for det in result.boxes:
                    cx1, cy1, cx2, cy2 = map(int, det.xyxy[0])
                    cv2.rectangle(cropped_img, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)

            # Save the resulting image
            cv2.imwrite(output_image_path, cropped_img)

if __name__ == "__main__":
    source = './video/test_mov'
    detect_objects_within_bounding_boxes(source, best_model)