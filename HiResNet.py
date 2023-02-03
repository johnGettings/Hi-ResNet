from keras.models import Model
from keras.layers import *
from keras.applications.resnet import ResNet50

from model_heads import *



def HiResNet(size, weights, classes):

    if size == 448:
        hi_res_head = resnet_448_head()
    elif size == 896:
        hi_res_head = resnet_896_head()
    elif size == 1792:
        hi_res_head = resnet_1792_head()
    else:
        raise ValueError('size should be an integer value of: 448, 896, or 1792')
    
    if not isinstance(classes, int):
        raise ValueError('classes must be an integer')
    
    if weights == "Res50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    elif weights == "None":
        base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError('weights should be either: \"Res50\" or \"None\"')

    # Constructing the ResNet50 base model. Removing top and first seven layers
    
    truncated_model = Model(inputs = base_model.layers[7].input, outputs = base_model.layers[-1].output)

    #Combining HiResNet head with ResNet50 base
    final_model = truncated_model(hi_res_head.output)
    model = Model(inputs=hi_res_head.input, outputs=final_model, name='HiResnet')

    # adding final layer
    head_model = MaxPool2D(pool_size=(4, 4))(model.output)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(1024, activation='relu')(head_model)
    head_model = Dropout(0.2)(head_model)
    head_model = Dense(512, activation='relu')(head_model)
    head_model = Dropout(0.2)(head_model)
    head_model = Dense(classes, activation='softmax')(head_model)

    # final configuration
    return Model(model.input, head_model)