import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,metrics

class LSTM(keras.Model):
    def __init__(self):
        super(LSTM,self).__init__()
        self.model = keras.Sequential([
        ])

    def call(self,inputs):
        x = self.model(inputs)
        return x

    def train_step(self,data):
        return loss