import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
photos_dir = 'photos1'
photos = map(lambda file_name: os.path.join(photos_dir, file_name), os.listdir(photos_dir))


def display_image(image, display_scale=0.4):
    height, width = image.shape[0], image.shape[1]
    image_display = cv2.resize(image, (int(display_scale * width), int(display_scale * height)))
    cv2.imshow('image', image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for photo_path in photos:
    img = cv2.imread(photo_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    display_image(img)
