from keras.layers import *
from tensorflow.keras.regularizers import l2
from keras.models import Model

def res_identity(x, filters):
    x_skip = x
    f1, f2 = filters

    #first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    #second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    # third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input
    x = Add()([x, x_skip])
    x = Activation(activation='relu')(x)

    return x

def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    # second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    #third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add
    x = Add()([x, x_skip])
    x = Activation(activation='relu')(x)

    return x

def resnet_1792_head(input_shape=(1792,1792,3)):

    input_im = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    x = Conv2D(8, kernel_size=(7, 7), strides=(2, 2))(x) #Output size 896
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x) #Output size 448

    x = res_conv(x, s=1, filters=(8, 32))
    x = res_identity(x, filters=(8, 32))
    x = res_identity(x, filters=(8, 32))

    x = res_conv(x, s=2, filters=(16, 64)) #Output Size 224
    x = res_identity(x, filters=(16, 64))
    x = res_identity(x, filters=(16, 64))
    x = res_identity(x, filters=(16, 64))

    x = res_conv(x, s=2, filters=(32, 128)) #Output size 112
    x = res_identity(x, filters=(32, 128))
    x = res_identity(x, filters=(32, 128))
    x = res_identity(x, filters=(32, 128))

    x = Conv2D(64, kernel_size=(1, 1), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    model = Model(inputs=input_im, outputs=x, name='HiResnet50')

    return model

def resnet_896_head(input_shape=(896,896,3)):

    input_im = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    x = Conv2D(16, kernel_size=(7, 7), strides=(2, 2))(x) #Output size 448
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x) #Output size 224

    x = res_conv(x, s=1, filters=(16, 64))
    x = res_identity(x, filters=(16, 64))
    x = res_identity(x, filters=(16, 64))

    x = res_conv(x, s=2, filters=(32, 128)) #Output size 112
    x = res_identity(x, filters=(32, 128))
    x = res_identity(x, filters=(32, 128))
    x = res_identity(x, filters=(32, 128))

    x = Conv2D(64, kernel_size=(1, 1), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    model = Model(inputs=input_im, outputs=x, name='HiResnet50-896')

    return model

def resnet_448_head(input_shape=(448,448,3)):

    input_im = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    x = Conv2D(32, kernel_size=(7, 7), strides=(2, 2))(x) #Output size 224
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x) #Output size 112

    x = res_conv(x, s=1, filters=(32, 128)) 
    x = res_identity(x, filters=(32, 128))
    x = res_identity(x, filters=(32, 128))
    x = res_identity(x, filters=(32, 128)) 

    x = Conv2D(64, kernel_size=(1, 1), strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001))(x) #Output size 56
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    model = Model(inputs=input_im, outputs=x, name='HiResnet50-448')

    return model
