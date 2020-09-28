import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, concatenate, GlobalAveragePooling2D

# def build_model(frame_size, action_dim, agent_history_length):
#     inputs = Input(shape=(frame_size, frame_size, agent_history_length))
#     conv1 = Conv2D(32, (8, 8), strides=8, activation='relu')(inputs)
#     conv2 = Conv2D(64, (4, 4), strides=2, activation='relu')(conv1)
#     conv3 = Conv2D(64, (3, 3), strides=1, activation='relu')(conv2)
#     flatten = Flatten()(conv3)
#     d1 = Dense(512, activation='relu')(flatten)
#     d2 = Dense(action_dim)(d1)
#     model = Model(inputs=inputs, outputs=d2)
#     return model

def build_model(frame_size, action_dim, agent_history_length):
    inputs = Input(shape=(frame_size, frame_size, agent_history_length))
    conv0_0 = Conv2D(32, (8, 8), strides=2, activation='selu')(inputs)
    conv0_1 = Conv2D(32, (3, 3), padding='same' ,activation='selu')(conv0_0)
    conv0_2 = Conv2D(32, (3, 3), padding='same', activation='selu')(conv0_1)
    concat0 = concatenate([conv0_0, conv0_2], axis=-1)
    conv1_0 = Conv2D(64, (3, 3), strides=2, activation='selu')(concat0)
    conv1_1 = Conv2D(64, (3, 3), padding='same', activation='selu')(conv1_0)
    conv1_2 = Conv2D(64, (3, 3), padding='same', activation='selu')(conv1_1)
    concat1 = concatenate([conv1_0, conv1_2], axis=-1)
    output_conv = Conv2D(action_dim, (1,1))(concat1)
    gap = GlobalAveragePooling2D()(output_conv)
    model = Model(inputs=inputs, outputs=gap)
    return model