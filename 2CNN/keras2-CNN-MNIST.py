# MNIST keras CNN model

import numpy as np
from keras.utils import np_utils

import matplotlib.pyplot as plt

path='mnist.npz'
f = np.load(path)
x_Train, y_Train = f['x_train'], f['y_train']
x_Test, y_Test = f['x_test'], f['y_test']
f.close()

np.random.seed(10)

print('train data: ', x_Train.shape, len(x_Train))
print('test data: ', x_Test.shape, len(x_Test))


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()

plot_image(x_Train[0])

# features(数字图像特征值）数据预处理
print('x_train_image: ', x_Train.shape)
print('y_train_label: ', y_Train.shape)
print('x_test_image: ', x_Test.shape)
print('y_test_label: ', y_Test.shape)

x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')
x_Test4D  = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')

# features 标准化
x_Train4D_normalize = x_Train4D / 225
x_Test4D_normalize = x_Test4D / 225

# label 用 One-Hot Encodeer
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot  = np_utils.to_categorical(y_Test)


# 建立模型
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters = 16,kernel_size = (5,5),padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid')) # 10 个输出

print(model.summary())

# 开始训练
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])

train_history = model.fit(x=x_Train4D_normalize,
                          y=y_TrainOneHot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=300,
                          verbose=2)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

show_train_history(train_history,'acc', 'val_acc')

show_train_history(train_history,'loss','val_loss')



# 评估模型准确率
scores = model.evaluate(x_Test4D_normalize, y_TestOneHot)
scores[1]

# 进行预测
prediction = model.predict_classes(x_Test4D_normalize)
prediction[:10]


def matplot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25
    for i in range(0, num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx], cmap='binary')
        title = 'label='+str(labels[idx])
        if len(prediction)>0:
            title+=', prediction='+str(prediction[idx])
            
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
    
matplot_images_labels_prediction(x_Test, y_Test, prediction, idx=10)


# 显示混淆矩阵

import pandas as pd

pd.crosstab(y_Test, prediction,
            rownames=['label'], colnames=['predict'])

























