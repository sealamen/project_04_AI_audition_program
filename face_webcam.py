
# 만약 캡쳐 화면이 안나올 경우 50번 라인 주석 풀고 실행했다가 다시 실행하면 됩니다.
import sys
import time
import cv2  # pip install opencv-python 설치
import numpy as np
from PIL import Image  # pillow로 설치
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model  # tensorflow 2.3버젼 필요
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import face_audition_rc

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

form_window = uic.loadUiType('./images/face_audition3.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model_01 = load_model('./models/male_and_female0.6365972.h5')
        self.model_02 = load_model('./models/age_gender.h5')
        self.model_03 = load_model('./models/emotion_predict_64model0.97979796.h5')
        self.btn_play.clicked.connect(self.image_open_slot)

    def image_open_slot(self):

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
        self.flag = True
        while self.flag:
            ret, frame = self.capture.read()
            # cv2.imshow('VideoFrame', frame) # 웹캠 창 안뜨게끔 주석처리.
            time.sleep(0.01)
            cv2.imwrite('./capture.png', frame)
            pixmap = QPixmap('./capture.png')
            self.lbl_image.setPixmap(pixmap)
            try:
                img= Image.open('./capture.png')
                img= img.convert('RGB')
                img_01 = img.resize((64, 64))
                img_02 = img.resize((48, 48))
                img_01 = np.expand_dims(img_01, axis=0)
                img_02 = np.expand_dims(img_02, axis=0)
                data_01 = np.asarray(img_01)
                data_02 = np.asarray(img_02)

            except:
                print('error')

            categories_kor = ['화내는', '누군가를 깔보는', '역겨워하는', '벌벌 떠는',
                              '기뻐하는', '슬퍼하는', '깜짝 놀라는']

            predict_value_01 = self.model_01.predict(data_01) # 성별 예측
            predict_value_02 = self.model_02.predict(data_02) # 나이 예측
            predict_value_03 = self.model_03.predict(data_01) # 감정 예측

            # # 성별예측 1: 주연님의 모델 사용
            # if predict_value_01 < 0.37:
            #     self.lbl_sex.setText('남자일 확률이'
            #         + str(((1 - predict_value_01[0][0])*100).round(1)) + '% 입니다')
            # else:
            #     self.lbl_sex.setText('여자일 확률이 '
            #         + str(((1-predict_value_01[0][0])*100).round(1)) + '% 입니다')

            # 성별예측 2: 현섭님의 모델 사용
            sex_f = ['Male', 'Female']
            self.lbl_sex.setText('성별은' + sex_f[int(np.round(predict_value_02[0][0]))] + '입니다.')

            # 나이 예측
            self.lbl_age.setText('나이는' + str((int(np.round(predict_value_02[1][0])))/100) + '살로 보입니다.')

            # 감정 예측
            self.lbl_emotion.setText('지금 ' + categories_kor[np.argmax(predict_value_03[-1])] + ' 장면을 연기하고 있습니다')


            # 웹캠 창 닫는 코드 : esc 눌러서 멈추면 닫기. 하지만 위에서 웹캠이 안나오게 했으므로 없어도 됩니다.

            key = cv2.waitKey(33)
            if key == 27:   # esc 버튼 누르고 닫으세요
                self.flag = False

    # 닫기버튼 누르면 닫히게 하는 함수(exam08 참조)
    def closeEvent(self, QCloseEvent):
        self.flag = False
        self.capture.release()
        QCloseEvent.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())

