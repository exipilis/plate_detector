from keras.layers import Conv2D, BatchNormalization, Dense, Input, Activation, Flatten
from keras.models import Model


def is_plate_model():
    input_image = Input((54, 128, 3))
    x = Conv2D(8, 3, activation='selu', padding='same')(input_image)
    x = Conv2D(8, 3, strides=2, activation='selu', padding='same')(x)
    x = Conv2D(8, 3, strides=2, activation='selu', padding='same')(x)
    x = Conv2D(8, 3, strides=2, activation='selu', padding='same')(x)
    x = Conv2D(8, 3, strides=2, activation='selu', padding='same')(x)
    x = Conv2D(8, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    m = Model(inputs=input_image, outputs=x)
    m.compile('adam', 'binary_crossentropy', ['accuracy'])

    return m
