from PIL import Image  # PIL은 pillow로 설치
import glob
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# CK+48 데이터셋 필요

img_dir = './CK+48/'
categories = ['anger', 'contempt', 'disgust', 'fear', 'happy',
              'sadness', 'surprise']
X = []
Y = []

# 이미지(X)를 숫자로 바꿈, 카테고리(Y)를 숫자로 바꿈
for idx, category in enumerate(categories):
    files = glob.glob(img_dir + category + '/*')
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((64, 64))
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
            # print(category, ':', f)
        except:
            print(category, i, '')


# 숫자를 0과 1로 변경
onehot_Y = to_categorical(Y)
print(X[980])
print(Y[980])
print(onehot_Y[980])
# print(len(X))
# print(len(Y))
print(len(onehot_Y))

X = np.array(X)
onehot_Y = np.array(onehot_Y)
X = X / 255
print(X[980])

X_train, X_test, Y_train, Y_test = train_test_split(X, onehot_Y,
                                                    test_size=0.1)
xy = (X_train, X_test, Y_train, Y_test)
np.save('./binary_image_data64.npy', xy)