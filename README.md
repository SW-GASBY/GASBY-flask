# flask

running YOLO

GASBY-flask-yolo<br>
├── app.py - Main API 파일.<br>
├── create_gifs.py - 생성된 데이터를 테스트 확인하기 위해 gif 생성.<br>
├── json_convert.py - 연결된 데이터를 통해 유저 별로 데이터 생성.<br>
├── player_tracking.py - 학습된 데이터를 통해 프레임 별 같은 사람으로 추정되는 바운딩 박스끼리 연결.<br>
└── video_handler.py - 입력된 비디오를 프레임 별로 분할하고, YOLO 모델을 통해 학습.<br>
