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
#     d2 = Dense(512, activation='relu')(flatten)
#     v = Dense(1)(d1)
#     a = Dense(action_dim)(d2)
#     outputs = v + (a - tf.reduce_mean(a))
#     model = Model(inputs=inputs, outputs=outputs)
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
    # Value function stream
    v_conv2_0 = Conv2D(256, (1,1), activation='selu')(concat1)
    v_conv2_1 = Conv2D(1, (1,1))(v_conv2_0)
    v_gap = GlobalAveragePooling2D()(v_conv2_1)
    # Avantage function stream
    a_conv2_0 = Conv2D(256, (1,1), activation='selu')(concat1)
    a_conv2_1 = Conv2D(action_dim, (1,1))(a_conv2_0)
    a_gap = GlobalAveragePooling2D()(a_conv2_1)
    outputs = v_gap + (a_gap - tf.reduce_mean(a_gap))
    model = Model(inputs=inputs, outputs=outputs)
    return model