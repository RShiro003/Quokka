
import cv2

# Ŭ���� �� (COCO ����, class ID 15�� 'person')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# ���� �Ʒõ� MobileNet-SSD �� �ε�
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt.txt',  # ���� ���� ����
    'MobileNetSSD_deploy.caffemodel'     # ����ġ ����
)

# ��ķ ����
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # �Է� �̹��� ��ó��
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # ���� ��� �ݺ�
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # �ŷڵ� �Ӱ谪
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] == "person":
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                # �ڽ� �� �� ���
                label = "���: {:.2f}%".format(confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ȭ�鿡 ���
    cv2.imshow("Person Detection", frame)

    # ���� ����
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ����
cap.release()
cv2.destroyAllWindows()
