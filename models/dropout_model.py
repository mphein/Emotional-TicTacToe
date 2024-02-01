from models.model import Model
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt

class DropoutModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        self.model = models.Sequential([
            layers.Conv2D(8, (1, 1), padding='same', activation='relu', input_shape=input_shape),
            #layers.Dropout(.1),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(16, (3,3), padding='same', activation='relu'),
            #layers.Dropout(.1),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            #layers.Dropout(.1),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            #layers.Dropout(.1),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (3,3), padding='same', activation='relu'),
            #layers.Dropout(.1),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(256, (5,5), padding='same', activation='relu'),
            #layers.Dropout(.1),
            layers.MaxPooling2D(pool_size=(2, 2)),


            
            layers.Flatten(),
            layers.Dropout(.5),
            layers.Dense(256, activation = 'relu'),

            layers.Dense(128, activation = 'relu'),
            
            #layers.Dropout(.1),
            layers.Dense(64, activation = 'relu'),
            layers.Dropout(.1),
            layers.Dense(32, activation = 'relu'),
            #layers.Dropout(.1),
            layers.Dense(categories_count, activation = 'softmax')
        ])
        # self.model = <model definition>
    
    def _compile_model(self):
        # Your code goes here
         self.model.compile(
            optimizer = keras.optimizers.legacy.Adam,
            loss= 'categorical_crossentropy',
            metrics= ['accuracy']
            )   
        # self.model.compile(<configuration properties>)