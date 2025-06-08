
import cv2

# 클래스 라벨 (COCO 기준, class ID 15가 'person')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# 사전 훈련된 MobileNet-SSD 모델 로드
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt.txt',  # 구조 정의 파일
    'MobileNetSSD_deploy.caffemodel'     # 가중치 파일
)

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 입력 이미지 전처리
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # 감지 결과 반복
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # 신뢰도 임계값
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # 박스 및 라벨 출력
                label = "사람: {:.2f}%".format(confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("Person Detection", frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
