import tensorflow.keras
import numpy as np
import cv2
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

cam = cv2.VideoCapture('img\\02.jpg') #ใส่ที่อยู่ของรูปที่ต้องการจะทำนายค่าความเป็นไปได้ตรงนี้

text = ""
percent = 0

while True:
    _,img = cam.read()
    img = cv2.resize(img,(224, 224))

    #turn the image into a numpy array
    image_array = np.asarray(img)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    # print(prediction)
    # จากโฟลเดอร์ img 01-03 คือ not (ธนวัฒน์)
    # จากโฟลเดอร์ img 11-13 คือ vi (พชร)
    # จากโฟลเดอร์ img 21-23 คือ mint (เพื่อนต่างมหาลัยที่ขอรูปมา เนื่องจากสมาชิกกลุ่มมีแค่สองคน)
    for i in prediction:
        if i[0] >= 0.9: #จากบรรทัดนี้ i[0] >= 0.9 หมายถึงค่าความเป็นไปได้ มากกว่าหรือเท่ากับ 90% ที่จะเป็นรูป not
            text ="not" 
            percent=i
        if i[1] >= 0.9: #จากบรรทัดนี้ i[1] >= 0.9 หมายถึงค่าความเป็นไปได้ มากกว่าหรือเท่ากับ 90% ที่จะเป็นรูป vi
            text ="vi"
            percent=i
        if i[2] >= 0.9: #จากบรรทัดนี้ i[2] >= 0.9 หมายถึงค่าความเป็นไปได้ มากกว่าหรือเท่ากับ 90% ที่จะเป็นรูป mint
            text ="mint"
            percent=i

       
        img = cv2.resize(img,(500, 500))  # resize รูป 
        cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1) # ใส่ text ลงไปในรูปที่จะแสดง
    cv2.imshow('img',img) # แสดงภาพหากมีความเป็นไปได้ 90% ขึ้นไป
    print("percent = ",percent) # แสดงค่าความเป็นไปได้ทั้งหมด ดังนี้ percent = (not, vi, mint)
    cv2.waitKey() # ทำให้ต้องรอการกดปุ่มใดปุ่มหนึ่งจากผู้ใช้ จึงจะหยุดแสดงภาพ และ สิ้นสุดการทำงานของโปรแกรม 
