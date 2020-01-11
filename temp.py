# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sun Nov 24 13:53:18 2019
# enviormnent : use python 3.7
# @author: zhanminxiang

# 遭遇問題 : 主要是環境、版本衝突、Python程式bug
# issue : ValueError: With n_samples=0, test_size=0.1 and train_size=None,
# the resulting train set will be empty. Adjust any of the aforementioned parameters.
# 原因：scikit-learn 版本太新
# 解法：把scikit-learn 降低版本到0.19.1

# issue : Running setup.py install for scikit-learn ... error
# 原因：Python 3.7 is not supported by scikit-learn 0.19.1.
# 解法：把環境改回3.6

# issue : 但遇到新的問題 import cv2 || ModuleNotFoundError: No module named 'cv2'
# 原因：cv2不支援3.6
# 解法：更新環境到3.7

# issue : model.save('h5/'+file_name+'.h5')顯示沒有目標路徑
# 解法 ： os.mkdir('h5/')新增路徑

# issue : model.save('h5/'+file_name+'.h5')顯示沒有目標路徑
# 解法 ： os.mkdir('h5/')新增路徑

# issue : model = load_model('h5/?_?.h5')找不到目標檔案
# 解法 : model = load_model('h5/'+file_name+'.h5')

# issue : FileExistsError: [Errno 17] File exists: 'h5/'
# 解法 : 先判斷目標地有無檔案夾

# 訓練時間超乎想像
# 其實原本就有心理準備這次訓練會很久了，但最終訓練的時間還是遠遠的超過自己當初的想像
# """

# <輸入>
# 匯入函式庫 -> 讀取圖片、標籤、名稱 -> 分割測試集、訓練集 

# <處理>
# 編譯模型、訓練 

# <輸出>
# 預測資料 -> 輸出預測值


# 匯入函式庫
import sys,os,cv2
import pandas as pd
import numpy as np
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout

# 定義參數
images = []
labels = []
name = []

pic_size = 64
batch_size = 32
epochs = 100
file_name = str(epochs)+'_'+str(batch_size)

main()

def main():
    # 讀取圖片、標籤
    images, labels = read_main(
        'machine-learningntut-classroom-2019-autumn/train/characters-20')

   

    # 分割測試集、訓練集、編譯模型、訓練
    train_compile_model()

    # 預測資料
    images = read_images('machine-learningntut-classroom-2019-autumn/test/test/')
    model = load_model('h5/'+file_name+'.h5')
    predict = model.predict_classes(images, verbose=1)
    print(predict)
    label_str = transform(np.loadtxt('name.txt', dtype='str'),
                        predict, images.shape[0])

    # save predict result 
    df = pd.DataFrame({"character": label_str})
    df.index = np.arange(1, len(df) + 1)
    df.index.names = ['id']
    df.to_csv('test.csv')

def train_compile_model():
     # 分割測試集、訓練集
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15)
    model = Sequential()
    compile_model(model)
    datagen = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True)
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=200, epochs=epochs,
                            validation_data=(X_test, y_test), verbose=1)

    # save model (check if there is an exist folder,and if not build one)
    if os.path.isdir('h5/'):
        model.save('h5/'+file_name+'.h5')
    else:
        os.mkdir('h5/')
        model.save('h5/'+file_name+'.h5')
    # score = model.evaluate (X_test, y_test, verbose=1)
    # print(score)

def compile_model(model):
    # convolution -> pooling -> dropout
    model.add(Conv2D(64, kernel_size=3 , padding='same',activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # convolution -> pooling -> dropout
    model.add(Conv2D(128, kernel_size=3, padding='same',activation='relu'))
    model.add(Conv2D(128, kernel_size=3, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # flatten -> activation -> dropout -> output
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20,activation='softmax'))
    model.summary()

    #optimimzer sgd 
    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])




def read_images(path):
    images = []
    for i in range(990):
        image = cv2.resize(cv2.imread(path+str(i+1)+'.jpg'), (64, 64))
        images.append(image)

    images = np.array(images, dtype=np.float32)/255
    return images


def transform(listdir, label, lenSIZE):
    label_str = []
    for i in range(lenSIZE):
        temp = listdir[label[i]]
        label_str.append(temp)

    return label_str

# 讀取圖片、標籤、名稱
def read_images_labels(path, i):
    #f = open('test.txt', 'w+')
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))
       # f.write(abs_path + '\n')
        if os.path.isdir(abs_path):
            i += 1
            temp = os.path.split(abs_path)[-1]
            name.append(temp)
            read_images_labels(abs_path, i)
            amount = int(len(os.listdir(path)))
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i) +
                             '[%s%%]' % (i*100/amount)+temp)

        if file.endswith('.jpg'):
            image = cv2.resize(cv2.imread(abs_path), (64, 64))
            images.append(image)
            labels.append(i-1)
            
    #f.close()
    return images, labels, name


def read_main(path):
    images, labels, name = read_images_labels(path, i=0)
    # print(images.shape)
    images = np.array(images, dtype=np.float32)/255
    labels = np_utils.to_categorical(labels, num_classes=20)
    np.savetxt('name.txt', name, delimiter=' ', fmt="%s")
    return images, labels



# ######################################################################
