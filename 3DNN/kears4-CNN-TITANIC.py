# 多层感知机预测泰坦尼克游客数据集

import urllib.request
import os

import matplotlib.pyplot as plt

url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath="data/titanic3.xls"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('download: ', result)

import numpy as np
import pandas as pd

all_df = pd.read_excel(filepath)

cols =['survived','name','pclass',  'sex', 'age', 'sibsp','parch','fare', 'embarked']
all_df=all_df[cols]

all_df[:2]



# 用drop方法删除name字段
df=all_df.drop(['name'], axis=1)

all_df.isnull().sum()

# 计算age平均值，然后将null值变为平均值
age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)

df['embarked'] = df['embarked'].fillna('C')

df.isnull().sum()


# 将文字转变为0与1
df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)

# 将embarked字段进行一位有效编码
x_OneHot_df = pd.get_dummies(data=df, columns=['embarked'])
x_OneHot_df[:2]

# 将 DataFrame 转化为 Array
ndarray = x_OneHot_df.values
ndarray.shape

ndarray[:2]

# 用 slice 语句提取features与label
label = ndarray[:,0]
features = ndarray[:,1:]

features[:2]
label[:5]


# 将 ndarray 特征字段进行标准化
from sklearn import preprocessing

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1)) 
scaled_features = minmax_scale.fit_transform(features)

scaled_features[:2]











# 将数据分成训练数据与测试数据

# 将数据以随机方式分为训练数据与测试数据
msk = np.random.rand(len(all_df)) < 0.8 # 8 : 2 比例

msk2 = np.empty([len(msk)], dtype=bool) # True->Fasle, False->True

index = 0
for i in msk:
    if i == True:
        msk2[index] = False
    else:
        msk2[index] = True
    index+=1

train_df = all_df[msk] 
test_df  = all_df[msk2]

print('total:', len(all_df),
      'train:', len(train_df),
      'test:',  len(test_df))


# 创建 PreprocessData 函数进行数据的预处理
def preprocess_data(raw_df):
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['embarked'] = df['embarked'].fillna('C')
    df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)
    df.isnull().sum()
    x_OneHot_df = pd.get_dummies(data=df, columns=['embarked'])
    
    ndarray = x_OneHot_df.values
    label = ndarray[:,0]
    features = ndarray[:,1:]
    
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1)) 
    scaled_features = minmax_scale.fit_transform(features)
    
    return scaled_features, label

train_features, train_label = preprocess_data(train_df)
test_features, test_label   = preprocess_data(test_df)

'''
# 打乱数据顺序
from sklearn.utils import shuffle

train_features2,train_label2 = shuffle(train_features,train_label)
test_features2,test_label2 = shuffle(test_features,test_label)
'''

# 建立模型
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(input_dim=9,
                units=40,
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=30,
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

train_history = model.fit(x=train_features,
                          y=train_label,
                          validation_split=0.1,
                          epochs=100,
                          batch_size=30,
                          verbose=1)


def show_train_history(train_history, train='acc', validation='val_acc'):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()
    
show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')


# 评估模型准确率
scores = model.evaluate(x=test_features, y=test_label)
print(scores[1])
print(scores[0])


# 加入Jack 和 Rose的数据
Jack = pd.Series([0, 'Jack', 3, 'male',   23, 1, 0, 5.0000,   'S'])
Rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])

# 创建 Pandas DataFramen JR_df, 加入jack和rose数据
JR_df = pd.DataFrame([list(Jack), list(Rose)],
                      columns = ['survived', 'name', 'pclass', 'sex',
                                'age', 'sibsp', 'parch', 'fare', 'embarked'])

all_df2=pd.concat([all_df,JR_df]) 

all_features, all_label = preprocess_data(all_df2)

all_probability = model.predict(all_features)

all_probability[-2:]


# 将 all_df 与 all_probability 整合
pd = all_df2
pd.insert(len(all_df2.columns),
          'probability', all_probability)
JR = pd[-2:]

moving = pd[(pd['survived']==0) & (pd['probability']>0.9)]























