# MNIST keras DNN model

# import libraries
import numpy as np
import pandas as pd
from keras.utils import np_utils

import matplotlib.pyplot as plt

path='mnist.npz'
f = np.load(path)
x_train_image, y_train_label = f['x_train'], f['y_train']
x_test_image, y_test_label = f['x_test'], f['y_test']
f.close()

np.random.seed(10)

print('train data: ', y_train_label.shape, len(x_train_image))
print('test data: ', x_test_image.shape, len(x_test_image))


def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()

plot_image(x_test_image[0])
y_test_label[0]


# to plot the 10 training data and thier labels
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

matplot_images_labels_prediction(x_train_image, y_train_label, [], 0, num=10)


# features(数字图像特征值）数据预处理
print('x_train_image: ', x_train_image.shape)
print('y_train_label: ', y_train_label.shape)
print('x_test_image: ', x_test_image.shape)
print('y_test_label: ', y_test_label.shape)

# reshape为一维向量， 再用astype转换为Float，共784个浮点数
x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')

print('x_Train: ', x_Train.shape)
print('x_Test: ', x_Test.shape)

# normalize
x_Train_normalize = x_Train/255
x_Test_normalize = x_Test/255


# label 数据预处理,用 One-Hot Encoding,转化为10个0或1的组合
y_train_label[:5]
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot  = np_utils.to_categorical(y_test_label)
y_TrainOneHot[:5]


# build the model， 共两层神经网络，隐藏层256个神经元，输出层10个神经元
from keras.models import Sequential
from keras.layers import Dense

input_num    = 784
hidden_layer = 1000 # hidden_layer=256: accuracy= 0.9756, hidden_layer=256: accuracy= 0.9789              
output_num   = 10
initializer = 'normal'
activation1 = 'relu'
activation2 = 'sigmoid'
loss_func = 'categorical_crossentropy'
optimizer = 'adam'
validation_split = 0.2
epochs = 30
batch_size = 200

model = Sequential()

model.add(Dense(input_dim = input_num,
                units   = hidden_layer,
                kernel_initializer = initializer,
                activation= activation1))

model.add(Dense(units     = output_num,
          kernel_initializer = initializer,
          activation= activation2))

print(model.summary()) # 看模型的摘要


# start training 
model.compile(loss=loss_func,
              optimizer = optimizer,
              metrics=['accuracy'])

train_histort = model.fit(x=x_Train_normalize,
                          y=y_TrainOneHot,
                          validation_split=validation_split, # 0.8作为训练数据，0.2作为验证数据
                          epochs=epochs,
                          batch_size=batch_size,       
                          verbose=2)            # 显示训练过程


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

show_train_history(train_histort, 'acc', 'val_acc')
# 训练准确率 比 验证的准确率高

show_train_history(train_histort, 'loss', 'val_loss')
# 在Epoch训练后期，’loss 训练的误差’ 比 ‘val_loss验证的误差’小

# 以测试数据评估模型准确率
scores = model.evaluate(x_Test_normalize, y_TestOneHot)
print('accuracy=',scores[1]) # accuracy= 0.9757

prediction = model.predict_classes(x_Test_normalize)

# 打印十张被预测的图片，标记 label 和 prediction
matplot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=250)


# 显示混淆矩阵
pd.crosstab(y_test_label, prediction, 
            rownames = ['label'], 
            colnames = ['predict'])


df = pd.DataFrame({'label':y_test_label,
                   'predict': prediction})

df[:2]

# 查询真实值是 5 ，但预测值是 3 的数据
df[(df.label==5)&(df.predict==3)]

# 查看340项结果
matplot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=340, num=1)

# using dropout
from keras.layers import Dropout

model2 = Sequential()

model2.add(Dense(input_dim = input_num,
                units   = hidden_layer,
                kernel_initializer = initializer,
                activation= activation1))

model2.add(Dropout(0.2))

model2.add(Dense(units     = output_num,
          kernel_initializer = initializer,
          activation= activation2))

print(model2.summary())

model2.compile(loss=loss_func,
              optimizer = optimizer,
              metrics=['accuracy'])

train_histort2 = model2.fit(x=x_Train_normalize,
                            y=y_TrainOneHot,
                            validation_split=validation_split, # 0.8作为训练数据，0.2作为验证数据
                            epochs=epochs,
                            batch_size=batch_size,       
                            verbose=2) 

show_train_history(train_histort, 'acc', 'val_acc')
show_train_history(train_histort2, 'acc', 'val_acc')

scores = model.evaluate(x_Test_normalize, y_TestOneHot) # hidden_layer = 1000
print('accuracy=',scores[1]) # accuracy= 0.9757

scores = model2.evaluate(x_Test_normalize, y_TestOneHot)
print('accuracy=',scores[1]) # accuracy= 0.9803


# 3-layer hidden_layers

model3 = Sequential()

model3.add(Dense(input_dim = input_num,
                units   = hidden_layer,
                kernel_initializer = initializer,
                activation= activation1))

model3.add(Dropout(0.5))

model3.add(Dense(units     = output_num,
          kernel_initializer = initializer,
          activation= activation1))

model3.add(Dropout(0.5))

model3.add(Dense(units     = output_num,
          kernel_initializer = initializer,
          activation= activation2))

print(model3.summary())

model3.compile(loss=loss_func,
              optimizer = optimizer,
              metrics=['accuracy'])

train_histort3 = model3.fit(x=x_Train_normalize,
                            y=y_TrainOneHot,
                            validation_split=validation_split, # 0.8作为训练数据，0.2作为验证数据
                            epochs=epochs,
                            batch_size=batch_size,       
                            verbose=2) 

show_train_history(train_histort, 'acc', 'val_acc')
show_train_history(train_histort2, 'acc', 'val_acc')

scores = model.evaluate(x_Test_normalize, y_TestOneHot) # hidden_layer = 1000
print('accuracy=',scores[1]) # accuracy = 0.9847

scores = model3.evaluate(x_Test_normalize, y_TestOneHot)
print('accuracy=',scores[1]) # accuracy= 0.9742































