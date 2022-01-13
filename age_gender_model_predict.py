# load model

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# 이미지 주기(이미지 경로 적으시면 됩니다)
img_file_dir = './CK+48/perfect-smile_bg.jpg'

categories = ['anger', 'contempt', 'disgust', 'fear', 'happy',
              'sadness', 'surprise']


# 받아온 이미지데이터 숫자로 바꿔주기
img = Image.open(img_file_dir)
img = img.convert("RGB")
img = img.resize((48, 48))
img = np.expand_dims(img, axis=0)
# 모델에 넣을 때 1, 48, 48, 3 이렇게 4차원이됨. 따라서 여기서도 차원을 하나 높여줘야 사이즈가 맞음
data = np.asarray(img)
print(data.shape)

# 모델이 예측하게끔 하자
model = load_model('./models/age_gender.h5')
model.summary()
preds = model.predict(data)
print(preds)

sex_f = ['Male', 'Female']
age = int(np.round(preds[1][0]))
sex = int(np.round(preds[0][0]))
age = age /100
print("Predicted Age: " + str(age))
print("Predicted Sex: " + sex_f[sex])



