<div align="center">

# BING: Flask Server for YOLOv8

*BING 서비스 객체 인식에 사용된 python code 및 API code 입니다.*

[![Static Badge](https://img.shields.io/badge/language-english-red)](./README.md) [![Static Badge](https://img.shields.io/badge/language-korean-blue)](./README-KR.md) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FSinging-voice-conversion%2Fsingtome-model&count_bg=%23E3E30F&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>

<br>

**SW중심대학 디지털 경진대회** : SW와 생성 AI의 만남 - SW 부문
팀 GASBY의 BING 서비스
이 리포지토리는 팀 GASBY가 SW중심대학 디지털 경진대회에서 개발한 BING 서비스에 사용된 Flask Server 함수의 코드를 포함하고 있습니다. 본 프로젝트는 생성 AI 기술을 활용하여 사용자의 요구에 맞는 다양한 소프트웨어 솔루션을 제공합니다.

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
| <a href="https://github.com/wooing1084"><img src="https://avatars.githubusercontent.com/u/32007781?v=4" height="120px"></a>| SungHoon Jung <br> **wooing1084**| Data collection and selection, model learning and management in the fields of court segmentation|

<br>


</div>

<br>

## 1. LAMBDA 소개

저희는 총 7개의 Lambda 함수를 사용합니다. 각 Lambda 함수는 특정 상황에 따라 동작하며, 각각의 트리거에 따라 작동하여 자동으로 파이프라인이 실행되도록 설계되었습니다. 이로 인해 데이터 처리 및 결과 생성 과정이 원활하고 효율적으로 이루어집니다.
## upload-gasby-request
- **Role**: 유저의 영상 업로드 및 요청
- **Endpoint**: https://nj7ceu0i9c.execute-api.ap-northeast2.amazonaws.com/deploy/request

- **Method**: Post
- Request Example:
    
    ```python
    import requests
    import base64
    
    url = 'https://nj7ceu0i9c.execute-api.ap-northeast-2.amazonaws.com/deploy/request'
    
    # user 요청받아서 만들기
    file_path = '/Users/jungheechan/Desktop/kakao.mp4'
    userId = '1'
    
    # 파일을 base64로 인코딩
    with open(file_path, 'rb') as f:
        file_content = f.read()
        file_content_base64 = base64.b64encode(file_content).decode('utf-8')
    
    # HTTP POST 요청 보내기
    payload = {
        'file': file_content_base64,
        'userId': userId  
    }
    
    response = requests.post(url, json=payload)
    
    # 응답 확인
    print(response.status_code)
    print(response.json())
    ```
    
- **Trigger: API Gateway**

## **run-mot**
- **Role**: 유저의 요청에 따른 MOT predict 실행
- **Endpoint**: Trigger로 작동

- **Trigger: S3- [gasby-req](https://ap-northeast-2.console.aws.amazon.com/s3/buckets/gasby-req?region=ap-northeast-2)**
- Response:
    
    ```python
    {
    'payload': userId
    }
    ```
    

## run-actrecog
- **Role**: 유저의 요청에 따른 action recognition predict 실행
- **Endpoint**: Trigger로 작동

- **Trigger: S3- [gasby-mot-result](https://ap-northeast-2.console.aws.amazon.com/s3/buckets/gasby-mot-result?region=ap-northeast-2)**
- Response:
    
    ```python
    {
    'payload': userId
    }
    ```
    

## mot-trigger
- **Role**: 새로운 pt 파일 생성시 MOT train 실행
- **Endpoint**: Trigger로 작동

- **Trigger: S3- [gasby-mot](https://ap-northeast-2.console.aws.amazon.com/s3/buckets/gasby-mot?region=ap-northeast-2)**
- Response:
    
    ```python
     payload = {
            'file_url': file_url
        }
    ```
    

## actrecog-trigger
- **Role**: 새로운 pt 파일 생성시 action recognition train 실행
- **Endpoint**: Trigger로 작동

- **Trigger: S3 -  [gasby-actrecog](https://ap-northeast-2.console.aws.amazon.com/s3/buckets/gasby-actrecog?region=ap-northeast-2)**
- Response:
    
    ```python
     payload = {
            'file_url': file_url
        }
    ```
    

### GPT
- **Role**: 최종 Action Recog 결과물(action.json)으로 해설 생성
- **Endpoint**: Trigger로 작동

- **Trigger: S3 -** [gasby-actrecog-result](https://ap-northeast-2.console.aws.amazon.com/s3/buckets/gasby-actrecog-result?region=ap-northeast-2)


## 2. Architecture

**LAMBDA** : S3에서 트리거를 받아 모델 예측 요청 및 결과값 S3에 저장


**s3** : 사용자의 입력, 각 모델의 결과를 저장하기 위한 저장소


**on-premise** : 모델 학습, 예측을 위한 GPU 엔드포인트 서버

<img width="852" alt="image" src="https://github.com/user-attachments/assets/b7d47d98-ddc3-4333-b090-e3cf7a575ec5">


## 3. Other AWS Resources

- **S3**:
<img width="1119" alt="Screenshot 2024-07-25 at 9 15 48 PM" src="https://github.com/user-attachments/assets/6a83c2a3-3f7f-4107-b51e-c53d72e06a97">


