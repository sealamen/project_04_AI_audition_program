# load model

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 이미지 주기(이미지 경로 적으시면 됩니다)
img_file_dir = './CK+48/pic2.jpg'


# 받아온 이미지데이터 숫자로 바꿔주기
img = Image.open(img_file_dir)
img = img.convert("RGB")
img = img.resize((64, 64))
img = np.expand_dims(img, axis=0)
# 모델에 넣을 때 1, 48, 48, 3 이렇게 4차원이됨. 따라서 여기서도 차원을 하나 높여줘야 사이즈가 맞음
data = np.asarray(img)
print(data.shape)

# 모델 예측
model = load_model('models/male_and_female0.6365972.h5')
model.summary()
preds = model.predict(data)
print(preds)

if preds < 0.5:
    print('남자일 확률이' + str(((1 - preds[0][0]) * 100).round()) + '% 입니다.')
else:
    print('여자일 확률이' + str(((preds[0][0]) * 100).round()) + '% 입니다.')
