import tensorflow as tf

###### Residual Block
class ResBlock(tf.keras.layers.Layer):

    """
    Residual Block: Conv2D + BN + ReLU + Residual Connection
    """

    def __init__(self,num_filters,kernel_size):
        
        #### Defining Essentials
        super().__init__()
        self.num_filters = num_filters # Number of filters
        self.kernel_size = kernel_size # Kernel Size of the Conv2D

        #### Defining Layers
        self.Conv2D = tf.keras.layers.Conv2D(filters=self.num_filters,
                                             kernel_size=self.kernel_size,
                                             activation='linear',padding='same')
        self.BN = tf.keras.layers.BatchNormalization()

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size
        })
        return config

    def call(self,X_in):

        """
        Residual Block: Conv2D + BN + ReLU + Residual Connection

        INPUTS:-
        1) X_in: Input to the residual block

        OUTPUTS:-
        1) X_res: Output of the residual block of same shape as that of input
        """

        X_res = self.Conv2D(X_in) # 2D-Convolution
        X_res = self.BN(X_res) # Batch-Normalization
        X_res = tf.keras.layers.Activation('relu')(X_res) # ReLU Activation
        X_res = tf.keras.layers.Add()([X_res,X_in]) # Residual Connection
        return X_res

###### ISCA Module
class ISCA(tf.keras.layers.Layer):

    """
    Layer to compute difference across frames and average them
    """

    def __init__(self,T):

        #### Defining Essentials
        super().__init__()
        self.T = T # Number of frames

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'T': self.T,
        })

        return config
    
    def call(self, X_in):

        """
        Sampling across T frames of the input

        INPUTS:-
        1) X_in: Input of shape (N,T,H,W,C)

        OUTPUTS:-
        1) X_o: ISCA output of shape (N,H,W,T-1)
        """

        ###### Extraction of Temporal Signals
        X_red_M1 = X_in[:,:-1,:,:,:] # Taking the X_red till the penultimate frame
        X_red_M2 = X_in[:,1:,:,:,:] # Taking the X_red from the second frame till the end
        X_diff = tf.keras.layers.Add()([X_red_M2,-X_red_M1]) # Computing difference between the frames

        ###### Channel Average and Permuation
        X_avg = tf.math.reduce_mean(X_diff,axis=-1,keepdims=False) # Averaging
        X_o = tf.keras.layers.Permute((2,3,1))(X_avg) # Permutation

        return X_o 
    
###### Behavioural Energy Fusion Module
class BE_Fusion(tf.keras.layers.Layer):

    """
    BE-Fusion Module to fuse features
    """

    def __init__(self,lambda_scale):

        #### Defining Essentials
        super().__init__()
        self.lambda_scale = lambda_scale # Scaling factor for Physiological Features

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'lambda_scale': self.lambda_scale # Lambda Scaling
        })
        return config

    def call(self,X_P,X_B):

        """
        BE-Fusion Modul fuse features

        INPUTS:
        1) X_P: Physiological Embedding of shape (N,d_p)
        2) X_B: Behavioural Embedding of shape (N,d_b)

        OUTPUTS:
        1) X_o: Scaled and Concatenated output of shape (N,d_p+d_b) 
        """

        #### Normalization of Physiological Embeddings
        def normalisation_layer(x):
            return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))        
        X_P = self.lambda_scale*tf.keras.layers.Lambda(normalisation_layer)(X_P) # Normalization

        #### Energy Computation
        X_BE_Scale = tf.math.reduce_sum(tf.square(X_B),axis=-1,keepdims=True) # Behavioural Energy
        X_BE_Scale = 1/(tf.sqrt(self.lambda_scale**2+X_BE_Scale)) # Scaling Factor

        #### Concatenation and Scaling
        X_o = tf.keras.layers.Concatenate(axis=-1)([X_P,X_B]) # Concatenation Operation
        X_o = tf.math.multiply(X_o,X_BE_Scale) # Multiplication Operation

        return X_o