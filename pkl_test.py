import pickle
import cv2

# 저장된 PKL 파일 경로
pkl_file_path = 'video/test_mov' + '/frames.pkl'

# PKL 파일 로드
with open(pkl_file_path, 'rb') as f:
    frames = pickle.load(f)

# 데이터 출력
for i, frame in enumerate(frames):
    # 프레임을 윈도우에 출력 (cv2.imshow를 사용하면 프레임을 창에 보여줄 수 있습니다)
    cv2.imshow(f'Frame {i}', frame)
    
    # 창을 닫기 위해 키 입력 대기
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 프레임 정보 출력 (shape 정보 예시)
    print(f'Frame {i}: shape = {frame.shape}')