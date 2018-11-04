# keras 5 - RNN - IMDb 自然语言处理

import urllib.request
import os
import tarfile


url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url, filepath)
    print('download:', result)
    

if not os.path.exists("data/acImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('data/')
    


# 读取 IMDb 数据
from keras.preprocessing import sequence        # 用于截长补短，‘数字列表’长度为100
from keras.preprocessing.text import Tokenizer  # 用于建立字典

# 用正则表达式删除 HTML 的标签
import re # 导入 Regulation Expression 模块

def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text) # 将符合正则表达式条件的字符串转化为空字符串


# 创建 read_files 函数读取 IMDb 文件 
import os

def read_files(filetype):
    path = "data/aclImdb"
    file_list = []
    
    positive_path = path + filetype+ "/pos/"
    for f in os.listdir(positive_path):
        file_list += [ positive_path + f ]
        
    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list +=[ negative_path + f ] 
        
    print('read', filetype, 'files:', len(file_list))
    
    all_labels = ([1]*12500+[0]*12500)     # list：前12500是1，后12500是0
    
    all_texts = []
    
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
    return all_labels, all_texts


# 读取 IMDb 数据集目录
y_train, train_text = read_files("/train")

y_test, test_text = read_files("/test")


# 建立 token 以及 token 的特性
token = Tokenizer(num_words=4000) # 建立一个有2000个单词的字典
token.fit_on_texts(train_text)

print(token.document_count)
print(token.word_index)


# 使用 token 将“影评文字” 转化成 “数字列表”
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)

print(train_text[0])
print(x_train_seq[0])


# 用 sequence.pad_sequences() 方法截长补短
x_train_100 = sequence.pad_sequences(x_train_seq, maxlen=400)
x_test_100  = sequence.pad_sequences(x_test_seq, maxlen=400)


# 加入嵌入层
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding

model = Sequential()

model.add(Embedding(input_dim=4000,
                    input_length=400,
                    output_dim=32))
model.add(Dropout(0.2))

# 建立多层感知机
model.add(Flatten())

model.add(Dense(units=256,
                activation='relu'))
model.add(Dropout(0.35))

model.add(Dense(units=1,
                activation='sigmoid'))

model.summary()


# 训练模型
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history = model.fit(x_train_100,y_train,batch_size=100,
                          epochs=10,verbose=2,validation_split=0.2)


# 评估模型准确率
scores = model.evaluate(x_test_100, y_test, verbose=1)
scores[0]
scores[1]
# 进行预测
predict = model.predict_classes(x_test_100)
predict[:10]

predict_classes = predict.reshape(-1)
predict_classes[:10]
len(predict_classes)

# 查看测试数据预测结果
SentimentDict = {1:'正面的', 2:'负面的'}

def display_test_sentiment(i):
    print(test_text[i])
    print('label真实值:', SentimentDict[y_test[i]],
          '预测结果:', SentimentDict[predict_classes[i]])

display_test_sentiment(6)



# 预测单独的 text 的情感分析
def predict_review(input_text):
    input_seq = token.texts_to_sequences([input_text])
    pad_input_seq = sequence.pad_sequences(input_seq, maxlen=400)
    predict_result = model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])
    
predict_review('''This movie is now my current Austen favorite. I've watched it 7 or 8 times so far. The acting, to my mind, is incredible. The way I notice good acting is when I find myself looking up from whatever I'm doing (sewing, though not my fingers together, hopefully, or boondoggling or whatever) in order to watch the character deliver his lines. It's the turn of expression, the cast of posture, that make the words come alive -- that's what makes good acting, as far as I'm concerned.Well, I watch almost every part of "Northanger Abbey" because almost all the actors play their roles with such charisma. Peter Firth is amazing as Mr. Tilney, the perfect blend of Bathian fop and real, masculine hero - you're not sure until the end whether he's after Catherine's money or not. I love his touch of (Welsh?) accent. Mr. and Mrs. Richards are charming: the combination of their behaviors - especially Mr. Richards' high voice, lending counterpoint to his wit and wisdom - makes them so real. General Tilney as the hard-hearted father who may possibly be a murderer is fascinating, too. And Captain Tilney, the grinning rake who is so clearly enjoying himself... and the moneygrubbing sister and brother whose names I can't currently remember - the two of them are so perfectly, at once, smart and smarmy.The other reason I love this adaptation is that it is the most romantic of all the Jane Austen adaptations. I know this was one of Austen's weak points (well, it is as far as I am concerned): even though all her novels are love stories, it's hard to feel that her heroes and heroines are really in love at the end. And if they're aren't really in love, then what's the point? All the other adaptations I've seen (other than the early Olivier/Garson one) have pretty cold-fish kisses at the end, if they kiss at all. I don't at all like sex in movies but it really is necessary to have a heartfelt kiss in the end. And the ending kiss in Northanger is a doozy.The over-the-top approach to costumes, music, and lighting work very well as far as I'm concerned. And the script is extremely clever - the way we are educated about Gothic romance, highlife in Bath, Cathy's normal country upbringing, etc., is very well done, as they usually are in BBC productions. Also, I like the part when the little black page does the cartwheels. And the Marchionesse, I think, was an entirely appropriate and very clever''')
predict_review('''is the most romantic of all the Jane Austen adaptations. I know this was one of Austen's weak points (well, it is as far as I am concerned): even though all her novels are love stories, it's hard to feel that her heroes and heroines are really in love at the end. And if they're aren't really in love, then what's the point? All the other adaptations I've seen (other than the early Olivier/Garson one) have pretty cold-fish kisses at the end, if they kiss at all. I don't at all like sex in movies but it really is necessary to have a heartfelt kiss in the end. And the ending kiss in Northanger is a doozy.The over-the-top approach to costumes, music, and lighting work very well as far ''')




# RNN 模型
from keras.layers.recurrent import SimpleRNN

model2 = Sequential()

model2.add(Embedding(input_dim=4000,
                    input_length=400,
                    output_dim=32))
model2.add(Dropout(0.2))

# 建立多层感知机
model2.add(SimpleRNN(units=20)) # 16: 0.83528, 10: 0.81, 

model2.add(Dense(units=256,
                activation='relu'))
model2.add(Dropout(0.35))

model2.add(Dense(units=1,
                activation='sigmoid'))

model2.summary()


model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history2 = model2.fit(x_train_100,y_train,batch_size=100,
                           epochs=10,verbose=2,validation_split=0.2)

# 评估模型准确率
scores2 = model2.evaluate(x_test_100, y_test, verbose=1)
print(scores2[0])
print(scores2[1])




# LSTM 模型
from keras.layers.recurrent import LSTM

model3 = Sequential()

model3.add(Embedding(input_dim=4000,
                    input_length=400,
                    output_dim=32))
model3.add(Dropout(0.2))

# 建立多层感知机
model3.add(LSTM(units=32))

model3.add(Dense(units=256,
                activation='relu'))
model3.add(Dropout(0.35))

model3.add(Dense(units=1,
                activation='sigmoid'))

model3.summary()


model3.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history3 = model3.fit(x_train_100,y_train,batch_size=100,
                           epochs=10,verbose=2,validation_split=0.2)

# 评估模型准确率
scores3 = model3.evaluate(x_test_100, y_test, verbose=1)
print(scores3[0])
print(scores3[1])




















 







