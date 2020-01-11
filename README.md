
### 北科大 機器學習 辛普森分類器

#### 作法說明
首先處理辨識機器學習要辨識的圖片種類，這邊採取的作法是自動查找資料夾底下的所有檔案，如果是圖片則加入圖片列表、如果是資料夾則取資料夾名稱加入標籤列表、以及整理一張所有人名的檔案

![image](https://github.com/MachineLearningNTUT2018/classification-NTUT104331030/blob/master/截圖%202020-01-06%20下午1.38.04.png)
#### 程式流程 
1.  輸入
	-  匯入函式庫 
	- 讀取圖片、標籤、名稱 
	- 分割測試集、訓練集 
2. 處理
	- 編譯模型
	- 訓練模型
3. 輸出
	- 使用模型預測資料 
	- 輸出預測資料

##### 匯入函式庫
```python
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
```
##### 讀取圖片、標籤
![image](https://github.com/MachineLearningNTUT2018/classification-NTUT104331030/blob/master/pic/截圖%202020-01-06%20下午1.55.25.png)
##### 分割測試集、訓練集 
```python
#使用sklearn.model_selection的函示
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15)
```
##### 編譯模型
```python
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
```
##### 訓練模型
```python
# 使用keras裡面自帶的ImageDataGenerator來進行圖片的資料預處理
datagen = ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,
											height_shift_range=0.05, horizontal_flip=True)
datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                            steps_per_epoch=200, epochs=epochs,
                            validation_data=(X_test, y_test), verbose=1)
```

##### 預測辛普森成員
```python
# 使用前面的讀取圖片、標籤、名稱函示讀取測試資料
images = read_images('machine-learningntut-classroom-2019-autumn/test/test/')
# 載入模型
model = load_model('h5/'+file_name+'.h5')
# 預測資料集
predict = model.predict_classes(images, verbose=1)
```
##### 輸出預測資料
```python
 # 把預測值（index）轉為名字（string）
label_str = transform(np.loadtxt('name.txt', dtype='str'),predict, images.shape[0])

 # 儲存預測結果
df = pd.DataFrame({"character": label_str})
df.index = np.arange(1, len(df) + 1)
df.index.names = ['id']
df.to_csv('test.csv')
```
####遭遇問題：

遭遇問題 : 主要是環境、版本衝突、Python程式bug

1. issue : ValueError: With n_samples=0, test_size=0.1 and train_size=None,the resulting train set will be empty. Adjust any of the aforementioned parameters.
原因：scikit-learn 版本太新
解法：把scikit-learn 降低版本到0.19.1

1. issue : Running setup.py install for scikit-learn ... error
原因：Python 3.7 is not supported by scikit-learn 0.19.1.
解法：把環境改回3.6

2. issue : 但遇到新的問題 import cv2 || ModuleNotFoundError: No module named 'cv2'
原因：cv2不支援3.6
解法：更新環境到3.7

3. issue : model.save('h5/'+file_name+'.h5')顯示沒有目標路徑
解法 ： os.mkdir('h5/')新增路徑

4. issue : model.save('h5/'+file_name+'.h5')顯示沒有目標路徑
解法 ： os.mkdir('h5/')新增路徑

5. issue : model = load_model('h5/?_?.h5')找不到目標檔案
解法 : model = load_model('h5/'+file_name+'.h5')

6. issue : FileExistsError: [Errno 17] File exists: 'h5/'
解法 : 先判斷目標地有無檔案夾

7. 訓練時間超乎想像
其實原本就有心理準備這次訓練會很久了，但最終訓練的時間還是遠遠的超過自己當初的想像
#### 分數
![image](https://github.com/MachineLearningNTUT2018/classification-NTUT104331030/blob/master/pic/截圖%202020-01-06%20下午3.14.37.png)
