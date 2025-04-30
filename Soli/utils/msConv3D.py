import tensorflow as tf

class msConv3D(tf.keras.layers.Layer):

    def __init__(self,C):
        super().__init__()
        self.C = C
        self.conv1 = tf.keras.layers.Conv3D(filters=C,kernel_size=(1,1,1),padding='same',activation='relu')
        self.conv2 = tf.keras.layers.Conv3D(filters=C,kernel_size=(3,3,3),padding='same',activation='relu')
        self.conv3 = tf.keras.layers.Conv3D(filters=C,kernel_size=(5,5,5),padding='same',activation='relu')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'C': self.C,
        })
        return config

    def call(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        xo = tf.keras.layers.Add()([x1,x2,x3])
        return xo