#%%
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('dataset/ALL_Dialogues_in_friends-2.csv')
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)
dataset = dataset.drop(['Unnamed: 0'], axis=1)

#Remove \n
dataset['Dialogue'] = dataset['Dialogue'].str.replace('\n', ' ')
data = np.array(dataset['Dialogue'])

print(dataset.head())
# %%
#Tokenize
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',num_words=5000)
tokenizer.fit_on_texts(data)
x = tokenizer.texts_to_sequences(data)
maximum = max([len(i) for i in x])
x = pad_sequences(x, padding='post',maxlen=maximum)

print(x[:5])
# %%
def create_data(array_sequences):
    x = []
    y = [] 
    for i in range(len(array_sequences)):
        x.append(array_sequences[i][:-1])
        y.append(array_sequences[i][1:])
    return np.array(x), np.array(y)

x, y = create_data(x)

#To tensor dataset
dataset = tf.data.Dataset.from_tensor_slices((x,y))
dataset = dataset.shuffle(10000).batch(64,drop_remainder=True)

print(x.shape, y.shape)
# %%
def lstm_model(vocab_size,embedding_dim,rnn_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim),
        tf.keras.layers.LSTM(rnn_units,return_sequences=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units,return_sequences=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)])
    return model

#print(tokenizer.word_index)
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
rnn_units = 256

model = lstm_model(vocab_size,embedding_dim,rnn_units)
model.summary()

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss)
model.fit(dataset,epochs=5)

# %%
def generate_text(model,start_string,num_generate=100):
    input_eval = [tokenizer.word_index[i.lower()] for i in start_string.split()]
    input_eval = tf.expand_dims(input_eval,0)
    text_generated = []
    temperature = 1.0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(tokenizer.index_word[predicted_id+1])
    return (start_string + ' '.join(text_generated))

print(generate_text(model,'I hate you',num_generate=100))
# %%
