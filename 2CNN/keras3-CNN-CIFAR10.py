# CIFAR10 - CNN - Model

# 下载 CIFAR-10 数据
from keras.datasets import cifar10
import numpy as np
np.random.seed(10)

(x_img_train, y_label_train),\
(x_img_test, y_label_test) = cifar10.load_data()

print('x_img_train:',x_img_train.shape, len(x_img_train))
print('y_label_train :',y_label_train.shape, len(y_label_train))

print('x_img_test :',x_img_test.shape, len(x_img_test))
print('y_label_test :', y_label_test.shape, len(y_label_test))

x_img_train[0]


# 定义 label_dict 字典
label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
              5:'dog',      6:'frog',       7:'horse',8:'ship',9:'truck'}

# 修改 plot_images_labels_prediction() 函数
import matplotlib.pyplot as plt 

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num=25
    for i in range(0, num):
        ax = plt.subplot(5,5,i+1)
        ax.imshow(images[idx+i], cmap='binary')
        
        title = str(i+idx)+','+label_dict[labels[i+idx][0]]
        if len(prediction)>0:
            title += '=>' + label_dict[prediction[i+idx]]
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx += 1
    plt.show()
        
plot_images_labels_prediction(x_img_train, y_label_train, [], 0)


# 将图像预处理

# 看第一张图像的第一个点
x_img_train[0][0][0] # [59, 62, 63] : 分别代表 RGB

# 将图像 数字标准化
x_img_train_normalize = x_img_train.astype('float32')/255.0
x_img_test_normalize  = x_img_test.astype('float32')/255.0

x_img_train_normalize[0][0][0]

# 对label进行数据预处理  One-Hot Encoding
from keras.utils import np_utils
y_label_train_onehot = np_utils.to_categorical(y_label_train)
y_label_test_onehot  = np_utils.to_categorical(y_label_test)

y_label_train_onehot[:5]


# 建立模型
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

model = Sequential()

model.add(Conv2D(input_shape=(32,32,3),
                 filters=32, 
                 kernel_size=(3,3),
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3), 
                 activation='relu',
                 padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(10, activation='sigmoid'))

print(model.summary())



# 进行训练

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics  = ['accuracy'])

train_history = model.fit(x_img_train_normalize, y_label_train_onehot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=128,
                          verbose=2)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()
    
show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss','val_loss')




# 评估模型的准确率
scores = model.evaluate(x_img_test_normalize,
                        y_label_test_onehot,
                        verbose=1) # 1 是显示过程
scores[1] # 0是loss， 1是acc


# 进行预测
prediction = model.predict_classes(x_img_test_normalize) # 这里return一个index

plot_images_labels_prediction(x_img_test, y_label_test, prediction, 800, 10)


# 查看预测概率

# 这里return (10000, 10)的矩阵，表示10项所预测的概率值都是多少
predicted_probability = model.predict(x_img_test_normalize)


def show_predicted_probability(y, prediction, x, predicted_probability,i):
    # print 第i个在test内的图像，label以及其预测值
    print('label:',label_dict[y[i][0]],
          'predict', label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x[i], (32,32,3)))
    plt.show()
    
    # 打印这个图像，其10项预测的各概率值
    for j in range(10):
        print(label_dict[j]+
              ' probability:%1.9f'%(predicted_probability[i][j]))
    
show_predicted_probability(y_label_test, prediction,x_img_test, predicted_probability, 0)






# 显示混淆矩阵
prediction.shape
y_label_test.shape

# 将y_label_test的二维矩阵变为一维矩阵
y_label_test.reshape(-1)

import pandas as pd
print(label_dict)
pd.crosstab(y_label_test.reshape(-1), prediction, 
            rownames=['label'],colnames=['predict'])





# 现在添加3次卷积运算

model2 = Sequential()

model2.add(Conv2D(input_shape=(32,32,3),
                 filters=32, 
                 kernel_size=(3,3),
                 activation='relu',
                 padding='same'))

model2.add(Dropout(rate=0.3))

model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(filters=64,
                 kernel_size=(3,3), 
                 activation='relu',
                 padding='same'))

model2.add(Dropout(rate=0.3))

model2.add(Conv2D(filters=128,
                 kernel_size=(3,3), 
                 activation='relu',
                 padding='same'))

model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Flatten())
model2.add(Dropout(rate=0.3))

model2.add(Dense(2500, activation='relu'))
model2.add(Dropout(rate=0.3))

model2.add(Dense(1250, activation='relu'))
model2.add(Dropout(rate=0.3))

model2.add(Dense(10, activation='sigmoid'))

print(model2.summary())

model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics  = ['accuracy'])


try: 
    model2.load_weights("SaveModel/cifarCnnModel.h5")
    print("load the model successfully! ")
except:
    print('model fails! train a new network')

train_history = model2.fit(x_img_train_normalize, y_label_train_onehot,
                           validation_split=0.2,
                           epochs=5,
                           batch_size=300,
                           verbose=1)

scores2 = model2.evaluate(x_img_test_normalize,
                          y_label_test_onehot,
                          verbose=1) # 1 是显示过程
scores2[1] # 0是loss， 1是acc

model2.save_weights("SaveModel/cifarCnnModel.h5")
print("Save model to disk")





















