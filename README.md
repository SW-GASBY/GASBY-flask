<div align="center">

# BING: Flask Server for YOLOv8

*BING 서비스 객체 인식에 사용된 python code 및 API code 입니다.*

[![Static Badge](https://img.shields.io/badge/language-english-red)](./README.md) [![Static Badge](https://img.shields.io/badge/language-korean-blue)](./README-KR.md) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FSinging-voice-conversion%2Fsingtome-model&count_bg=%23E3E30F&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

<br>

**SW중심대학 디지털 경진대회** : SW와 생성 AI의 만남 - SW 부문
팀 GASBY의 BING 서비스
이 리포지토리는 팀 GASBY가 SW중심대학 디지털 경진대회에서 개발한 BING 서비스에 사용된 Flask Server 함수의 코드를 포함하고 있습니다. 본 프로젝트는 생성 AI 기술을 활용하여 사용자의 요구에 맞는 다양한 소프트웨어 솔루션을 제공합니다.

**레포지토리 개요**: 
BING 서비스는 최신 생성 AI 알고리즘을 사용하여 실시간으로 데이터를 처리하고 사용자에게 맞춤형 결과를 제공합니다. 이 프로젝트는 YOLOv8을 사용해 객체 인식 및 경기장 segmentation, 선수의 유니품 색상 classification을 위해서 사용합니다.

**주요 기능**: 
프레임 분할 및 객체 인식: YOLOv8 모델을 사용하여 학습한 파일을 이용하여 경기 영상을 프레임 별로 분할한 뒤에 프레임 내 선수, 심판, 공, 네트 객체들을 인식합니다.
경기장 코트 segmentation: YOLOv8-seg 모델을 사용하여 학습한 파일을 이용하여 농구 코트를 학습한 뒤에 코트 내에 있는 객체만 인식할 수 있도록 반영했습니다.
유니폼 색상 분류: YOLOv8-cls 모델을 사용하여 학습한 파일을 이용하여 프레임 내 선수들의 유니폼 색상을 파악합니다.

<br>

<div align="center">

<h3> SERVICE part Team members </h3>

| Profile | Name | Role |
| :---: | :---: | :---: |
| <a href="https://github.com/hspark-1"><img src="https://avatars.githubusercontent.com/u/105943940?v=4" height="120px"></a> | Hyunseo Park <br> **hspark-1**| YOLO object detecting & Uniform Color classify <br> Data collection and selection, model learning and management in the fields of detection and classify|
| <a href="https://github.com/wooing1084"><img src="https://avatars.githubusercontent.com/u/32007781?v=4" height="120px"></a>| SungHoon Jung <br> **wooing1084**| Data collection and selection, model learning and management in the fields of court segmentation|

<br>


</div>

<br>

## 1. Sever Code 소개

저희는 총 4개의 py파일을 사용합니다. 
## app.py
- **Role**: 유저가 비디오를 업로드 했을 때 aws trigger에서 요청을 보내는 엔드포인트입니다. s3에서 source를 다운로드한 뒤에 로직을 거쳐 선수와 공의 위치에 대한 파일을 업로드합니다.
- **Endpoint**: http://hostIP:port/yolo-predict/upload
- **Method**: Post

## video_handler.py
- **Role**: 영상을 프레임별로 분할한 뒤에 YOLOv8 model을 사용하여 객체를 인식하여 data.json 파일을 생성합니다.

## player_tracking.py
- **Role**: data.json 파일을 이용해서 프레임별 선수들의 bbox IoU를 계산하여 프레임별 선수의 위치를 예측 및 연결하여 tracked_results.json 파일과 ball.json 파일을 생성합니다.

## json_convert.py
- **Role**: tracked_results.json 파일을 이용하여 프레임 별로 구성되어있는 선수들의 정보를 같은 선수로 분류된 정보끼리 묶어서 json 파일을 변환하여 최종적으로 업로드할 파일을 생성합니다.

## 2. 실행 방법
1. .env파일을 생성하고 아래와같이 작성합니다.
```
AWS_Accesskey= 제공한 AWS 엑세스 키
AWS_Secretkey= 제공한 AWS 비밀 키
AWS_Region= 제공한 AWS 리전 정보
```
2. 터미널에 flask run --port 5000 입력하거나 파이썬 디버거를 활용해 app.py를 실행한다.
![image](https://github.com/user-attachments/assets/aaa023e1-9a79-4dcf-b038-c9a8f7c620bb)
