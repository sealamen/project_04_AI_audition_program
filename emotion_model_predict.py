# load model

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
img = img.resize((64, 64))
img = np.expand_dims(img, axis=0)
# 모델에 넣을 때 1, 48, 48, 3 이렇게 4차원이됨. 따라서 여기서도 차원을 하나 높여줘야 사이즈가 맞음
data = np.asarray(img)
print(data.shape)

# 모델 예측
model = load_model('models/emotion_predict_64model0.97979796.h5')
model.summary()
preds = model.predict(data)
print(preds)
# 7개의 수치를 보여줌
print(categories[np.argmax(preds[-1])])
# 제일 큰 애의 인덱스 값을 보여줌


