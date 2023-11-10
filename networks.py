"""
this python file contains all the created models
CNN class
    -> end_to_end_CNN model proposed by
            Hauke Dose, Jakob S. Møller, Helle K. Iversenb, Sadasivan Puthusserypadya
            An end-to-end deep learning approach to MI-EEG signal classification for BCIs (2018)

EEGNetModels class
    -> EEGNet model proposed by
            Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon, Chou P. Hung, Brent J. Lance
            EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces (2018)
    -> two_branch_EEGNet
    -> three_branch_EEGNet
    -> four_branch_EEGNet

    *parameters are adjusted based on models proposed by Ghadir Ali Altuwaijri and Ghulam Muhammad in
    Electroencephalogram-Based Motor Imagery Signals Classification Using a Multi-Branch Convolutional Neural Network
    Model with Attention Blocks and Nikoletta Nagy's observations*

EEGNetModelsCBAM class
    -> CBAM_simple just the CBAM attention module
    -> CBAM_EEGNet: EEGNet followed by a CBAM block
    -> CBAM_EEGNet_three_branch (FMBEEGCBAM) model proposed by
            Ghadir Ali Altuwaijri and Ghulam Muhammad in
            Electroencephalogram-Based Motor Imagery Signals Classification
            Using a Multi-Branch Convolutional Neural Network Model with Attention Blocks (2022)
    -> simple_CBAM_EEGNet_three_branch (MBEEGCBAM) model proposed by
            Ghadir Ali Altuwaijri and Ghulam Muhammad in
            Electroencephalogram-Based Motor Imagery Signals Classification
            Using a Multi-Branch Convolutional Neural Network Model with Attention Blocks (2022)

DSCNNModels class
    -> DSCNN model proposed by
            Weifeng Ma, Yifei Gong, Haojie Xue, Yang Liu, Xuefen Lin, GongxueZhou, Yaru Li in
            A lightweight and accurate double-branch neural network for four-class motor imagery classification (2022)
            (85% on BCI competition IV 2a)
    -> DSCNN_without_reshape
            same as DSCNN but accepts two inputs instead of using the Reshape layer

    *parameters are adjusted to the PhysioNet EEG Motor Movement/Imagery Dataset (2009) dataset*
"""

import numpy as np
from tensorflow import keras
from keras.constraints import max_norm
from keras.activations import elu, relu
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Concatenate, Reshape, LSTM, Multiply
from keras.layers import Conv1D, Conv2D, AveragePooling2D, AveragePooling3D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import Input, Flatten

from attention_modules import cbam, AttentionModuleCTAM, TemporalAttention, FiltersAttention, ChannelAttentionEEG


class Network:
    def __init__(self, batch_size=16, ep=10):
        self.batch_size = batch_size
        self.ep = ep
        self.model = keras.Sequential()
        self.build_model()

    def build_model(self):
        pass

    def train(self, x_train, train_label, x_valid, valid_label):
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
        self.model.summary()
        self.model.fit(x_train, train_label, batch_size=self.batch_size, epochs=self.ep, verbose=1,
                       validation_data=(x_valid, valid_label))

    def evaluate(self, x_test, y_test_one_hot):
        test_loss, test_acc = self.model.evaluate(x_test, y_test_one_hot, verbose=1)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        return test_loss, test_acc

    def predict(self, x_test):
        predicted_classes = self.model.predict(x_test)
        predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
        return predicted_classes


class ShallowConvNet(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):  # first model
        self.model = keras.Sequential([
            Conv2D(40, kernel_size=(1, 30), activation='relu', input_shape=(64, 321, 1), padding='same'),
            Conv2D(40, kernel_size=(64, 1), activation='relu', padding='valid'),
            AveragePooling2D(pool_size=(1, 15), padding='valid'),
            Flatten(),
            Dense(80, activation='relu'),
            Dense(4, activation='softmax'),
        ])


class EEGNet(Network):
    def __init__(self, batch_size=16, ep=10, f1=8, d=2, f2=16, samples=321, kern_len=160, channels=64, p=0.25):
        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.samples = samples
        self.kern_len = kern_len  # samples/2
        self.channels = channels
        self.p = p
        super().__init__(batch_size, ep)

    def build_model(self):  # EEGNet
        self.model = keras.Sequential([
            Conv2D(self.f1, kernel_size=(1, self.kern_len), use_bias=False, activation='linear',
                                input_shape=(self.channels, self.samples, 1), padding='same'),
            BatchNormalization(),
            DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear',
                                         padding='valid', depth_multiplier=self.d, depthwise_constraint=max_norm(1.)),
            BatchNormalization(),
            Activation(elu),
            AveragePooling2D(pool_size=(1, 4)),
            Dropout(self.p),  # default 0.25: cross-subject classification (change to 0.5 if within-subject classification)

            SeparableConv2D(self.f2, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same'),
            BatchNormalization(),
            Activation(elu),
            AveragePooling2D(pool_size=(1, 8)),
            Dropout(self.p),
            Flatten(),
            Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))
        ])


class DSCNN(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):
        input_left = Input((64, 321, 1))
        left = Conv2D(48, kernel_size=(1, 25), strides=(1, 1), activation='elu', input_shape=(64, 321, 1))(input_left)
        left = Conv2D(64, kernel_size=(64, 1), strides=(1, 1), activation='elu')(left)

        input_right = Reshape((1, 321, 64))(input_left)
        right = Conv1D(32, kernel_size=16, strides=1, activation='elu', input_shape=(1, 321, 64))(input_right)
        right = SeparableConv2D(64, kernel_size=(1, 25), strides=(1, 1), activation='elu',
                                depthwise_constraint=max_norm(1))(right)
        right = AveragePooling2D(pool_size=(1, 180), strides=(1, 30))(right)

        concatenated = Concatenate(axis=2)([left, right])
        concatenated = AveragePooling2D(pool_size=(1, 180), strides=(1, 15))(concatenated)

        flatten = Flatten()(concatenated)
        softmax_out = Dense(4, activation='softmax')(flatten)

        self.model = Model(inputs=input_left, outputs=softmax_out)
        return self.model


class TwoBranchEEGNet(Network):
    def __init__(self, batch_size=16, ep=10, d=2, channels=64):
        self.d = d
        self.channels = channels
        super().__init__(batch_size, ep)

    def build_model(self):  # two branch EEGNet
        input_main = Input((64, 321, 1))
        branch1 = Conv2D(4, kernel_size=(1, 16), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch1 = BatchNormalization()(branch1)
        branch1 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 4))(branch1)

        branch1 = SeparableConv2D(8, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 8))(branch1)
        # --------------------------------------------------------------------------------------------------------------
        branch2 = Conv2D(8, kernel_size=(1, 32), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch2 = BatchNormalization()(branch2)
        branch2 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 4))(branch2)
        branch2 = Dropout(0.1)(branch2)

        branch2 = SeparableConv2D(16, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 8))(branch2)
        branch2 = Dropout(0.1)(branch2)

        concatenated = Concatenate()([branch1, branch2])
        flatten_out = Flatten()(concatenated)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class ThreeBranchEEGNet(Network):
    def __init__(self, batch_size=16, ep=10, d=2, channels=64):
        self.d = d
        self.channels = channels
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        branch1 = Conv2D(4, kernel_size=(1, 16), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch1 = BatchNormalization()(branch1)
        branch1 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 4))(branch1)

        branch1 = SeparableConv2D(8, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 8))(branch1)
        # --------------------------------------------------------------------------------------------------------------
        branch2 = Conv2D(8, kernel_size=(1, 32), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch2 = BatchNormalization()(branch2)
        branch2 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 4))(branch2)
        branch2 = Dropout(0.1)(branch2)

        branch2 = SeparableConv2D(16, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 8))(branch2)
        branch2 = Dropout(0.1)(branch2)
        # --------------------------------------------------------------------------------------------------------------
        branch3 = Conv2D(16, kernel_size=(1, 64), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch3 = BatchNormalization()(branch3)
        branch3 = keras.layers.DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear',
                                               padding='valid', depth_multiplier=self.d,
                                               depthwise_constraint=max_norm(1))(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation(elu)(branch3)
        branch3 = AveragePooling2D(pool_size=(1, 4))(branch3)
        branch3 = Dropout(0.2)(branch3)

        branch3 = SeparableConv2D(32, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation(elu)(branch3)
        branch3 = AveragePooling2D(pool_size=(1, 8))(branch3)
        branch3 = Dropout(0.2)(branch3)
        # --------------------------------------------------------------------------------------------------------------
        concatenated = Concatenate()([branch1, branch2, branch3])
        flatten_out = Flatten()(concatenated)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class FourBranchEEGNet(Network):
    def __init__(self, batch_size=16, ep=10, d=2, channels=64):
        self.d = d
        self.channels = channels
        super().__init__(batch_size, ep)

    def build_model(self):  # four branch EEGNet
        input_main = Input((64, 321, 1))
        branch1 = Conv2D(4, kernel_size=(1, 16), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch1 = BatchNormalization()(branch1)
        branch1 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 4))(branch1)

        branch1 = SeparableConv2D(8, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 8))(branch1)

        # --------------------------------------------------------------------------------------------------------------
        branch2 = Conv2D(8, kernel_size=(1, 32), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch2 = BatchNormalization()(branch2)
        branch2 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 4))(branch2)
        branch2 = Dropout(0.1)(branch2)

        branch2 = SeparableConv2D(16, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 8))(branch2)
        branch2 = Dropout(0.1)(branch2)

        # --------------------------------------------------------------------------------------------------------------
        branch3 = Conv2D(16, kernel_size=(1, 64), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch3 = BatchNormalization()(branch3)
        branch3 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation(elu)(branch3)
        branch3 = AveragePooling2D(pool_size=(1, 4))(branch3)
        branch3 = Dropout(0.2)(branch3)

        branch3 = SeparableConv2D(32, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation(elu)(branch3)
        branch3 = AveragePooling2D(pool_size=(1, 8))(branch3)
        branch3 = Dropout(0.2)(branch3)
        # --------------------------------------------------------------------------------------------------------------
        branch4 = Conv2D(32, kernel_size=(1, 128), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch4 = BatchNormalization()(branch4)
        branch4 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch4)
        branch4 = BatchNormalization()(branch4)
        branch4 = Activation(elu)(branch4)
        branch4 = AveragePooling2D(pool_size=(1, 4))(branch4)
        branch4 = Dropout(0.4)(branch4)

        branch4 = SeparableConv2D(64, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch4)
        branch4 = BatchNormalization()(branch4)
        branch4 = Activation(elu)(branch4)
        branch4 = AveragePooling2D(pool_size=(1, 8))(branch4)
        branch4 = Dropout(0.4)(branch4)
        # --------------------------------------------------------------------------------------------------------------
        concatenated = Concatenate()([branch1, branch2, branch3, branch4])
        flatten_out = Flatten()(concatenated)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class MBEEGCBAM(Network):
    def __init__(self, batch_size=16, ep=10, channels=64):
        self.channels = channels
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        branch1 = Conv2D(4, kernel_size=(1, 16), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch1 = BatchNormalization()(branch1)
        branch1 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=2, depthwise_constraint=max_norm(1))(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 4))(branch1)

        branch1 = SeparableConv2D(8, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 8))(branch1)

        cbam1 = cbam(branch1, ratio=2, kernel_size=2)
        flatten1 = Flatten()(cbam1)

        # --------------------------------------------------------------------------------------------------------------
        branch2 = Conv2D(8, kernel_size=(1, 32), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch2 = BatchNormalization()(branch2)
        branch2 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=2, depthwise_constraint=max_norm(1))(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 4))(branch2)
        branch2 = Dropout(0.1)(branch2)

        branch2 = SeparableConv2D(16, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 8))(branch2)
        branch2 = Dropout(0.1)(branch2)

        cbam2 = cbam(branch2, ratio=8, kernel_size=4)
        flatten2 = Flatten()(cbam2)
        # --------------------------------------------------------------------------------------------------------------
        branch3 = Conv2D(16, kernel_size=(1, 64), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch3 = BatchNormalization()(branch3)
        branch3 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=2, depthwise_constraint=max_norm(1))(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation(elu)(branch3)
        branch3 = AveragePooling2D(pool_size=(1, 4))(branch3)
        branch3 = Dropout(0.2)(branch3)

        branch3 = SeparableConv2D(32, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation(elu)(branch3)
        branch3 = AveragePooling2D(pool_size=(1, 8))(branch3)
        branch3 = Dropout(0.2)(branch3)

        cbam3 = cbam(branch3, ratio=8, kernel_size=2)
        flatten3 = Flatten()(cbam3)
        # --------------------------------------------------------------------------------------------------------------

        flatten_out = Concatenate()([flatten1, flatten2, flatten3])
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class EEGNetCBAM(Network):
    def __init__(self, batch_size=16, ep=10, f1=8, d=2, f2=16, samples=321, kern_len=160, channels=64, p=0.25):
        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.samples = samples
        self.kern_len = kern_len  # sampling_rate/2
        self.channels = channels
        self.p = p
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        x = Conv2D(self.f1, kernel_size=(1, self.kern_len), use_bias=False, activation='linear',
                            input_shape=(self.channels, self.samples, 1), padding='same')(input_main)
        x = BatchNormalization()(x)
        x = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear',
                                     padding='valid', depth_multiplier=self.d, depthwise_constraint=max_norm(1.))(x)
        x = BatchNormalization()(x)
        x = Activation(elu)(x)
        x = AveragePooling2D(pool_size=(1, 4))(x)  # reduces the sampling rate
        x = Dropout(self.p)(x)  # default 0.25: cross-subject classification (change to 0.5 if within-subject classification)
        x = cbam(x, ratio=2, kernel_size=2)
        x = SeparableConv2D(self.f2, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(elu)(x)
        x = AveragePooling2D(pool_size=(1, 8))(x)
        x = Dropout(self.p)(x)

        flatten_out = Flatten()(x)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class ProposedNetwork(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        main = Conv2D(16, kernel_size=(1, 30), use_bias=True, activation='relu', input_shape=(64, 321, 1),
                      padding='same')(input_main)

        branch1 = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                  padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)

        branch2 = TemporalAttention(main, 8)
        branch2 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2)

        branch3 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(main)

        main = Concatenate()([branch1, branch2, branch3])
        main = FiltersAttention(main, 8)
        main = AveragePooling2D(pool_size=(1, 15))(main)

        flatten_out = Flatten()(main)
        flatten_out = Dense(80, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class ProposedNetworkEEGNet(Network):
    def __init__(self, batch_size=16, ep=10, channels=64, samples=321):
        self.channels = channels
        self.samples = samples
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        main = Conv2D(16, kernel_size=(1, 30), use_bias=True, activation='relu', input_shape=(64, 321, 1),
                      padding='same')(input_main)

        branch1 = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                  padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)

        branch2 = TemporalAttention(main, 8)
        branch2 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2)

        branch3 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(main)

        main = Concatenate()([branch1, branch2, branch3])
        main = FiltersAttention(main, 8)
        main = AveragePooling2D(pool_size=(1, 15))(main)
        main = Flatten()(main)
        # -------------------------------------------------------------------------------------------------------------
        eeg = Conv2D(8, kernel_size=(1, 80), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        eeg = BatchNormalization()(eeg)
        eeg = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=2, depthwise_constraint=max_norm(1))(eeg)
        eeg = BatchNormalization()(eeg)
        eeg = Activation(elu)(eeg)
        eeg = AveragePooling2D(pool_size=(1, 4))(eeg)
        eeg = Dropout(0.25)(eeg)

        eeg = SeparableConv2D(16, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(eeg)
        eeg = BatchNormalization()(eeg)
        eeg = Activation(elu)(eeg)
        eeg = AveragePooling2D(pool_size=(1, 8))(eeg)
        eeg = Dropout(0.25)(eeg)
        eeg = Flatten()(eeg)

        flatten_out = Concatenate()([main, eeg])
        # flatten_out = Flatten()(main)
        flatten_out = Dense(80, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class ProposedNetworkLSTM(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        main = Conv2D(16, kernel_size=(1, 30), use_bias=True, activation='relu', input_shape=(64, 321, 1),
                      padding='same')(input_main)

        branch1 = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                  padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)

        branch2 = TemporalAttention(main, 8)
        branch2 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2)

        branch3 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(main)

        main = Concatenate()([branch1, branch2, branch3])
        main = FiltersAttention(main, 8)

        main = Reshape((321, 64))(main)
        main = LSTM(units=40, input_shape=(321, 1), return_sequences=True)(main)
        main = LSTM(units=20, return_sequences=True)(main)
        main = LSTM(units=4, return_sequences=True)(main)
        main = Flatten()(main)
        softmax_out = Dense(4, activation='softmax')(main)

        self.model = Model(inputs=input_main, outputs=
        softmax_out)
        return self.model


class NewNetwork3(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        main = Conv2D(16, kernel_size=(1, 80), use_bias=True, activation='relu', input_shape=(64, 321, 1), padding='same')(input_main)

        branch1 = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                     padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)

        branch2 = TemporalAttention(main, 8)
        branch2 = Conv2D(32, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2)

        main = Concatenate()([branch1, branch2])
        main = AveragePooling2D(pool_size=(1, 4))(main)

        flatten_out = Flatten()(main)
        flatten_out = Dense(80, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class NewNetwork5(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        main = Conv2D(16, kernel_size=(1, 80), use_bias=True, activation='relu', input_shape=(64, 321, 1), padding='same')(input_main)
        branch1 = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                     padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)

        branch2 = TemporalAttention(main, 8)
        branch2 = Conv2D(32, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2)

        main = Concatenate()([branch1, branch2])
        main = AveragePooling2D(pool_size=(1, 4))(main)

        branch4 = FiltersAttention(main)
        branch4 = Conv2D(128, kernel_size=(1, 4), strides=(1, 4), use_bias=True, activation='relu', padding='valid')(branch4)

        branch5 = SeparableConv2D(128, kernel_size=(1, 8), use_bias=True, activation='relu', padding='same')(main)
        branch5 = AveragePooling2D(pool_size=(1, 4))(branch5)

        main = Concatenate()([branch4, branch5])

        flatten_out = Flatten()(main)
        flatten_out = Dense(80, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class NewNetwork8(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        main = Conv2D(16, kernel_size=(1, 30), use_bias=True, activation='relu', input_shape=(64, 321, 1), padding='same')(input_main)

        branch1 = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                     padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)

        branch2 = TemporalAttention(main, 8)
        branch2 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2)

        branch3 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(main)

        main = Concatenate()([branch1, branch2, branch3])
        main = AveragePooling2D(pool_size=(1, 15))(main)

        flatten_out = Flatten()(main)
        flatten_out = Dense(80, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class NewNetwork16(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        # --------------------------------------------------------------------------------------------------------------
        main = Conv2D(16, kernel_size=(1, 30), use_bias=True, activation='relu', input_shape=(64, 321, 1),
                      padding='same')(input_main)

        branch1 = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                  padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)

        branch2 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(main)
        branch2 = TemporalAttention(branch2, 8)

        branch3 = SeparableConv2D(32, kernel_size=(1, 15), use_bias=True, activation='relu',
                                     padding='same', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)
        branch3 = Conv2D(64, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch3)

        main = Concatenate()([branch1, branch2, branch3])
        main = FiltersAttention(main, 8)
        main = AveragePooling2D(pool_size=(1, 15))(main)

        flatten_out = Flatten()(main)
        flatten_out = Dense(80, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class NewNetwork17(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        main_left = Conv2D(16, kernel_size=(1, 30), use_bias=True, activation='relu', input_shape=(64, 321, 1),
                      padding='same')(input_main)

        branch1_left = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                  padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main_left)

        branch2_left = TemporalAttention(main_left, 8)
        branch2_left = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2_left)

        branch3_left = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(main_left)
        # --------------------------------------------------------------------------------------------------------------

        main_right = Conv2D(16, kernel_size=(1, 60), use_bias=True, activation='relu', input_shape=(64, 321, 1),
                           padding='same')(input_main)

        branch1_right = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                  padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main_right)

        branch2_right = TemporalAttention(main_right, 16)
        branch2_right = Conv2D(32, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2_right)
        # --------------------------------------------------------------------------------------------------------------

        main = Concatenate()([branch1_left, branch2_left, branch3_left, branch1_right, branch2_right])
        main = FiltersAttention(main, 8)
        main = AveragePooling2D(pool_size=(1, 15))(main)

        flatten_out = Flatten()(main)
        flatten_out = Dense(80, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class NewNetwork(Network):
    def __init__(self, batch_size=16, ep=10, f1=8, d=2, f2=16, samples=321, kern_len=80, channels=64, p=0.25):
        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.samples = samples
        self.kern_len = kern_len
        self.channels = channels
        self.p = p
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        main = Conv2D(16, kernel_size=(1, 30), use_bias=True, activation='relu', input_shape=(64, 321, 1),
                      padding='same')(input_main)

        branch1 = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                  padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main)
        branch1 = FiltersAttention(branch1, 8)

        branch2 = TemporalAttention(main, 8)
        branch2 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2)
        branch2 = FiltersAttention(branch2, 4)

        branch3 = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(main)
        branch3 = FiltersAttention(branch3, 4)

        main = Concatenate()([branch1, branch2, branch3])

        main = Reshape((321, 64))(main)
        main = LSTM(units=40, input_shape=(321, 1), return_sequences=True)(main)
        # main = Dropout(0.4)(main)
        # main = BatchNormalization()(main)
        main = LSTM(units=20, return_sequences=True)(main)
        main = LSTM(units=4, return_sequences=True)(main)
        # main = Dropout(0.4)(main)
        # main = BatchNormalization()(main)

        # main = AveragePooling2D(pool_size=(1, 15))(main)

        flatten_out = Flatten()(main)
        # flatten_out = Dense(20, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax')(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class EEGNetCBAMLikeAttentionModule(Network):
    def __init__(self, batch_size=16, ep=10, f1=8, d=2, f2=16, samples=321, kern_len=160, channels=64, p=0.25):
        self.f1 = f1
        self.d = d
        self.f2 = f2
        self.samples = samples
        self.kern_len = kern_len
        self.channels = channels
        self.p = p
        super().__init__(batch_size, ep)

    def build_model(self):
        input_main = Input((64, 321, 1))
        x = AttentionModuleCTAM(input_main)
        x = keras.layers.Conv2D(self.f1, kernel_size=(1, self.kern_len), use_bias=False, activation='linear',
                                input_shape=(self.channels, self.samples, 1), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear',
                                         padding='valid', depth_multiplier=self.d, depthwise_constraint=max_norm(1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(elu)(x)
        x = keras.layers.AveragePooling2D(pool_size=(1, 4))(x)  # reduce the sampling rate of the signal to 32 Hz
        x = keras.layers.Dropout(self.p)(x)
        # default 0.25: cross-subject classification (change to 0.5 if within-subject classification)

        x = keras.layers.SeparableConv2D(self.f2, kernel_size=(1, 16), use_bias=False, activation='linear',
                                         padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(elu)(x)
        x = keras.layers.AveragePooling2D(pool_size=(1, 8))(x)
        x = keras.layers.Dropout(self.p)(x)

        out = keras.layers.Flatten()(x)
        out = keras.layers.Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(out)

        self.model = Model(inputs=input_main, outputs=out)
        return self.model


class СBAMEEGNetThreeBranch(Network):
    def __init__(self, batch_size=16, ep=10, d=2, channels=64):
        self.d = d
        self.channels = channels
        super().__init__(batch_size, ep)

    def build_model(self):  # this is the one that I used, two concatenates: one after the EEGNet branches, one after СBAMs are added
        input_main = Input((64, 321, 1))
        branch1 = Conv2D(4, kernel_size=(1, 16), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch1 = BatchNormalization()(branch1)
        branch1 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 4))(branch1)

        branch1 = SeparableConv2D(8, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation(elu)(branch1)
        branch1 = AveragePooling2D(pool_size=(1, 8))(branch1)

        cbam1 = cbam(branch1, ratio=2, kernel_size=2)

        # --------------------------------------------------------------------------------------------------------------
        branch2 = Conv2D(8, kernel_size=(1, 32), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch2 = BatchNormalization()(branch2)
        branch2 = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear', padding='valid',
                                  depth_multiplier=self.d, depthwise_constraint=max_norm(1))(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 4))(branch2)
        branch2 = Dropout(0.1)(branch2)

        branch2 = SeparableConv2D(16, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation(elu)(branch2)
        branch2 = AveragePooling2D(pool_size=(1, 8))(branch2)
        branch2 = Dropout(0.1)(branch2)

        cbam2 = cbam(branch2, ratio=8, kernel_size=4)
        # --------------------------------------------------------------------------------------------------------------
        branch3 = Conv2D(16, kernel_size=(1, 64), use_bias=False, activation='linear', input_shape=(64, 321, 1),
                         padding='same')(input_main)
        branch3 = BatchNormalization()(branch3)
        branch3 = keras.layers.DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear',
                                               padding='valid', depth_multiplier=self.d,
                                               depthwise_constraint=max_norm(1))(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation(elu)(branch3)
        branch3 = AveragePooling2D(pool_size=(1, 4))(branch3)
        branch3 = Dropout(0.2)(branch3)

        branch3 = SeparableConv2D(32, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation(elu)(branch3)
        branch3 = AveragePooling2D(pool_size=(1, 8))(branch3)
        branch3 = Dropout(0.2)(branch3)

        cbam3 = cbam(branch3, ratio=8, kernel_size=2)
        # --------------------------------------------------------------------------------------------------------------
        concatenated = Concatenate()([branch1, branch2, branch3])
        flatten_1 = Flatten()(concatenated)

        cbamconcatenated = Concatenate()([cbam1, cbam2, cbam3])
        flatten_2 = Flatten()(cbamconcatenated)

        flatten_out = Concatenate()([flatten_1, flatten_2])
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


class ModifiedShallowConvNet(Network):
    def __init__(self, batch_size=16, ep=10):
        super().__init__(batch_size, ep)


    def build_model(self):
        input_main = Input((64, 321, 1))
        x = Conv2D(32, kernel_size=(1, 30), activation='relu', input_shape=(64, 321, 1), padding='same')(input_main)
        x = Conv2D(32, kernel_size=(64, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(1, 15), padding='valid')(x)

        x = Conv2D(48, kernel_size=(1, 7), activation='relu', padding='same')(x)

        # x = Dropout(0.4)(x)
        x = Flatten()(x)
        x = Dense(80, activation='relu')(x)
        softmax_out = Dense(4, activation='softmax')(x)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model