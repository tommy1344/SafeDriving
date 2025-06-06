import cv2
import uuid
import os
import time

IMAGES_PATH = os.path.join('data','images')
labels = ['awake','drowsy']
number_imgs = 20

cap = cv2.VideoCapture(0)
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(2)

    img_count = 0
    start_time = time.time()

    # collects images and assigns them unique names according to class, identifier, and adds jpg
    while img_count < number_imgs:
        ret,frame = cap.read()
        cv2.imshow('Image Collection', frame)
        current_time = time.time()
        if current_time - start_time >= 2:
            img_count += 1
            print('Collecting images for {}, image number {}'.format(label, img_count))
            imgname = os.path.join(IMAGES_PATH, label + '.' + str(uuid.uuid1())+'.jpg')
            cv2.imwrite(imgname,frame)
            start_time = current_time

        if cv2.waitKey(10) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows

# USE THIS TO LABEL THE IMAGES ACCORDINGLY
# !cd labelImg
# !python labelImg.py

# use this to train the data when all is done
# cd yolov5 && python train.py --img 320 --batch 16 --epochs 250 --data dataset.yml --weights yolov5s.pt --workers 2