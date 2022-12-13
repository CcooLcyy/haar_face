import cv2

STEP = 3

# 读取画面
img = cv2.imread('a.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 导入模型
face_detector = cv2.CascadeClassifier('./weight/haarcascade_frontalface_default.xml')

# 调整minNeighbors确保可以探测有且只有一个人脸
inspection_quality = 40
detections = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=inspection_quality)
while True:
    while len(detections) != 1:
        if(len(detections) < 1):
            inspection_quality = inspection_quality - STEP
            detections = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=inspection_quality)
        elif(len(detections) > 1):
            inspection_quality = inspection_quality + (STEP + 2)
            detections = face_detector.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=inspection_quality)
    while len(detections) == 1:
        for x,y,w,h in detections:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))

        while True:
            key = cv2.waitKey(0)
            cv2.imshow('img', img)
            if key == ord('q'):
                exit()