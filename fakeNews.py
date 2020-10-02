# %%
import pandas as pd
df = pd.read_csv('fakenews/train.csv')

df.head()
# %%

df = df.dropna()
# %%
x = df.drop('label',axis=1)

y = pd.DataFrame(df['label'])

import tensorflow as tf

from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot


voc_size = 5000
messages = x.copy()
messages.reset_index(inplace=True)


import nltk
import re
from nltk.corpus import stopwords


nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []


messages['title'][0]
#print(messages['title'][53])
#messages = messages.dropna()
for i in range(len(messages)):
    if type(messages['title'][i]) != str:
        print(i)
    else:
        print('all str')
# %%
for i in range(len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]',' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# %%
onehot_repr = [one_hot(words,voc_size) for words in corpus]
onehot_repr
# %%
# len(onehot_repr[0])
# lens = []
# for i in range(len(onehot_repr)):
#     lens.append(len(onehot_repr[i]))
# max(lens)
# %%
sent_len = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre',maxlen=sent_len)
print(embedded_docs)
# %%
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_len))
model.add(LSTM(100)) #one layer with 100 neurons
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
# %%
import numpy as np
# %%
x_final = np.array(embedded_docs)
y_final = np.array(y)
# %%
x_final.shape, y_final.shape
# %%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_final,y_final, test_size = 0.3, random_state=0)

# %%
history = model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=25,batch_size=64)
# %%
ypred= model.predict(xtest)
ypred = ypred > 0.5
# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest,ypred)
# %%
acc = accuracy_score(ytest,ypred)
# %%
