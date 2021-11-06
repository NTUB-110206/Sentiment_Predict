import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# 載入資料集
data = pd.read_csv('Sentiment.csv')
data = data[['text','sentiment']]

# 資料預處理
data = data[data.sentiment != "Neutral"]
data_pos = data[data.sentiment != "Negative"]
data_neg = data[data.sentiment != "Positive"]

data_neg = data_neg.sample(n = 2236)
data = data[data.sentiment == "Positive"]
data = data.append(data_neg, ignore_index=True)

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

# 預備模型
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# 訓練模型
Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

batch_size = 64
model.fit(X_train, Y_train, epochs = 8, batch_size=batch_size, verbose = 3)

# 儲存模型
now = datetime.now()
date_time_str = now.strftime("%Y%m%d%H%M%S")
model.save("senti_" + date_time_str + ".h5")