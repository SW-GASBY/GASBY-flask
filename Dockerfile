# 베이스 이미지로 Python 사용
FROM python:3.11

# 작업 디렉터리 설정
WORKDIR /app

# 필요한 파일들을 컨테이너에 복사
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY train8 train8

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
RUN pip install opencv-python

# 컨테이너 시작 시 실행될 명령어
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
