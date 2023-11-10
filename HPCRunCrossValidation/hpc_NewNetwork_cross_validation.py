import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from sklearn import model_selection as ms
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
from tensorflow import keras
from keras.constraints import max_norm
from keras.models import Model
from keras.layers import AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization, Dropout
from keras.layers import Input, Flatten
from keras.activations import elu
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import statistics

from tensorflow import reduce_mean, expand_dims, reduce_max
from keras.layers import Dense, Activation, Concatenate, Multiply, Reshape
from keras.layers import Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D


def visualize(y_test, predicted_classes, network_name=None, fold=None):
    cm = confusion_matrix(y_test, predicted_classes)
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, cmap='crest', fmt="d")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    if network_name is None:
        plt.show()


def ChannelAttention(input, ratio=8):  # x  = input feature map
    """
    Same architecture as the Channel Attention module of the CBAM but the dimensions are changed so that
    the function chooses the channels of the EEG signals
    :param input: input data
    :param ratio: reduction ratio
    :return: input with weighted channels
    """
    bm, channels, timesteps, filters = input.shape
    # Shared layers
    l1 = Dense(channels//ratio, activation='relu', use_bias=False)
    l2 = Dense(channels, use_bias=False)

    # Global Average Pooling
    x1 = GlobalAveragePooling2D(data_format='channels_first')(input)
    x1 = l1(x1)
    x1 = l2(x1)

    # Global Max Pool
    x2 = GlobalMaxPooling2D(data_format='channels_first')(input)
    x2 = l1(x2)
    x2 = l2(x2)

    out = x1 + x2

    out = Activation("sigmoid")(out)
    input_reshape = Reshape((filters, timesteps, channels))(input)

    out = Multiply()([input_reshape, out])
    out = Reshape((channels, timesteps, filters))(out)

    return out


def TemporalAttention(input, kernel_size=7):  # input  = input feature map
    """
    Same architecture as the Spatial Attention of the CBAM module, but the dimensions are changed, so that the function
    chooses the timesteps of interest
    :param input: input EEG signals
    :param kernel_size: size of the Conv2D layer's kernel
    :return: weighted input
    """
    # Average Pooling
    x1 = reduce_mean(input, axis=1)
    x1 = expand_dims(x1, axis=1)

    # Max Pooling
    x2 = reduce_max(input, axis=1)
    x2 = expand_dims(x2, axis=1)

    # Concatenate
    out = Concatenate()([x1, x2])

    # Conv layer
    out = Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')(out)
    out = Multiply()([input, out])

    return out


def AttentionModuleCBAM(x, ratio=16, kernel_size=7):
    x = ChannelAttention(x, ratio)
    x = TemporalAttention(x, kernel_size)
    return x


# --------------------------------------------------------------------------------------------------------------------


def FiltersAttention(input, ratio=8):  # x  = input feature map CHANNEL ATTENTION OF A CBAM
    """
    The Channel Attention Module of the CBAM, the name has been changed to FiltersAttention, because the dimensions of
    the EEG signals are not the same as the dimensions of an image. In this study, the last dimension, is the dimension
    of the filters
    :param input: input EEG signals
    :param ratio: reduction ratio
    :return: weighted input
    """
    bm, _, _, filters = input.shape
    # Shared layers
    l1 = Dense(filters//ratio, activation='relu', use_bias=False)
    l2 = Dense(filters, use_bias=False)

    # Global Average Pooling
    x1 = GlobalAveragePooling2D(data_format='channels_last')(input)
    x1 = l1(x1)
    x1 = l2(x1)

    # Global Max Pool
    x2 = GlobalMaxPooling2D(data_format='channels_last')(input)
    x2 = l1(x2)
    x2 = l2(x2)

    out = x1 + x2
    out = Activation("sigmoid")(out)
    out = Multiply()([input, out])

    return out


def SpatialAttention(input, kernel_size=7):  # x  = input feature map SPATIAL ATTENTION OF A CBAM
    """
    The Spatial Attention Module of the CBAM
    :param input: EEG signals
    :param kernel_size: size of the Conv2D layer's kernel
    :return: weighted input
    """
    # Average Pooling
    x1 = reduce_mean(input, axis=-1)
    x1 = expand_dims(x1, axis=-1)

    # Max Pooling
    x2 = reduce_max(input, axis=-1)
    x2 = expand_dims(x2, axis=-1)

    # Concatenate
    out = Concatenate()([x1, x2])

    # Conv layer
    out = Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')(out)
    out = Multiply()([input, out])

    return out


def cbam(x, ratio=8, kernel_size=7):
    x = FiltersAttention(x, ratio)
    x = SpatialAttention(x, kernel_size)
    return x


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
        self.model.fit(x_train, train_label, batch_size=self.batch_size, epochs=self.ep, verbose=2,
                       validation_data=(x_valid, valid_label))

    def evaluate(self, x_test, y_test_one_hot):
        test_loss, test_acc = self.model.evaluate(x_test, y_test_one_hot, verbose=2)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        return test_loss, test_acc

    def predict(self, x_test):
        predicted_classes = self.model.predict(x_test)
        predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

        return predicted_classes



class NewNetwork(Network):
    def __init__(self, batch_size=16, ep=10, f1=8, d=2, f2=16, samples=321, kern_len=80, channels=64, p=0.25):
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
        main_left = Conv2D(16, kernel_size=(1, 30), use_bias=True, activation='relu', input_shape=(64, 321, 1),
                      padding='same')(input_main)

        branch1_left = DepthwiseConv2D(kernel_size=(64, 1), use_bias=True, activation='relu',
                                  padding='valid', depth_multiplier=2, depthwise_constraint=max_norm(1.))(main_left)
        # branch1_left = FiltersAttention(branch1_left, 8)

        branch2_left = TemporalAttention(main_left, 8)
        branch2_left = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(branch2_left)
        # branch2_left = FiltersAttention(branch2_left, 4)

        branch3_left = Conv2D(16, kernel_size=(64, 1), use_bias=True, activation='relu', padding='valid')(main_left)
        # branch3_left = FiltersAttention(branch3_left, 4)

        main_left = Concatenate()([branch1_left, branch2_left, branch3_left])
        main_left = AveragePooling2D(pool_size=(1, 15))(main_left)
        main_left = Flatten()(main_left)
        # -------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        main_right = Conv2D(self.f1, kernel_size=(1, self.kern_len), use_bias=False, activation='linear',
               input_shape=(self.channels, self.samples, 1), padding='same')(input_main)
        main_right = DepthwiseConv2D(kernel_size=(self.channels, 1), use_bias=False, activation='linear',
                        padding='valid', depth_multiplier=self.d, depthwise_constraint=max_norm(1.))(main_right)
        main_right = Activation(elu)(main_right)
        main_right = AveragePooling2D(pool_size=(1, 4))(main_right)

        main_right = SeparableConv2D(self.f2, kernel_size=(1, 16), use_bias=False, activation='linear', padding='same')(main_right)
        main_right = BatchNormalization()(main_right)
        main_right = Activation(elu)(main_right)
        # main_right = FiltersAttention(main_right, 4)
        main_right = AveragePooling2D(pool_size=(1, 8))(main_right)
        main_right = Dropout(self.p)(main_right)
        main_right = Flatten()(main_right)
        # -------------------------------------------------------------------------------------------------------------
        main = Concatenate()([main_left, main_right])

        flatten_out = Flatten()(main)
        flatten_out = Dense(80, activation='relu')(flatten_out)
        softmax_out = Dense(4, activation='softmax', kernel_constraint=max_norm(0.25))(flatten_out)

        self.model = Model(inputs=input_main, outputs=softmax_out)
        return self.model


def create_sets(x_train, y_train, x_test, y_test):
    for i in range(x_train.shape[0]):  # normalizing the train data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_train[i])
        x_train[i] = scaler.transform(x_train[i])

    for i in range(x_test.shape[0]):  # normalizing the test data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_test[i])
        x_test[i] = scaler.transform(x_test[i])

    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)  # reshaping the train data
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)  # reshaping the test data

    y_train_one_hot = to_categorical(y_train)
    test_label = to_categorical(y_test)

    x_train, x_valid, train_label, valid_label = ms.train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=0)  # train and validate

    return x_train, train_label, x_valid, valid_label, x_test, test_label, y_test


def load_cross_validation(n_folds):
    runs = [4, 6, 8, 10, 12, 14]  # use only hand and feet motor imagery runs

    x = []  # contains the recordings by subject, so this is a list consisting of 105 elements
    y = []

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for subject in range(1, 110):
        if subject != 88 and subject != 89 and subject != 92 and subject != 100:
            save_to = 'datasets'
            eegbci.load_data(subject, runs, save_to, verbose=0)  # downloading the files

            x_tmp = []  # contains the recordings of one subject
            y_tmp = []

            for record in runs:
                s = '{:0>3}'.format(subject)
                r = '{:0>2}'.format(record)
                filepath = f'datasets/MNE-eegbci-data/files/eegmmidb/1.0.0/S{s}/S{s}R{r}.edf'
                raw = read_raw_edf(filepath, verbose=0)  # getting raw data

                event_dict_raw = {
                    'T0': 0,
                    'T1': 1,
                    'T2': 2
                }
                events_tmp, event_ids_tmp = mne.events_from_annotations(raw, event_id=event_dict_raw, verbose=0)  # events and their ids
                events_tmp = mne.pick_events(events_tmp, exclude=0)  # excluding rest, rest_id=0
                old_annotations = mne.annotations_from_events(events_tmp, 160)

                if record == 4 or record == 8 or record == 12:
                    annotation_keys_even = {
                        '1': 'left',
                        '2': 'right'
                    }

                    event_dict = {
                        'left': 0,
                        'right': 1
                    }
                    new_annotations = old_annotations.rename(mapping=annotation_keys_even)
                    raw.set_annotations(new_annotations)
                    events, event_ids = mne.events_from_annotations(raw, event_id=event_dict, verbose=0)

                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9, verbose=0)
                    epoch.equalize_event_counts()

                if record == 6 or record == 10 or record == 14:
                    annotation_keys_odd = {
                        '1': 'fists',
                        '2': 'feet'
                    }

                    event_dict = {
                        'fists': 2,
                        'feet': 3
                    }
                    new_annotations = old_annotations.rename(mapping=annotation_keys_odd)
                    raw.set_annotations(new_annotations)
                    events, event_ids = mne.events_from_annotations(raw, event_id=event_dict, verbose=0)

                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9, verbose=0)
                    epoch.equalize_event_counts()

                x_tmp.append(epoch.get_data())
                y_tmp.append(epoch.events[:, 2])

            x_tmp = np.concatenate(x_tmp)  # the recordings of one subject are concatenated
            y_tmp = np.concatenate(y_tmp)

            x.append(x_tmp)
            y.append(y_tmp)

    kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)

    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train_tmp = []
        y_train_tmp = []

        x_test_tmp = []
        y_test_tmp = []

        for index in train_index:
            x_train_tmp.append(x[index])
            y_train_tmp.append(y[index])

        for index in test_index:
            x_test_tmp.append(x[index])
            y_test_tmp.append(y[index])

        x_train.append(np.concatenate(x_train_tmp, axis=0))
        y_train.append(np.concatenate(y_train_tmp))

        x_test.append(np.concatenate(x_test_tmp, axis=0))
        y_test.append(np.concatenate(y_test_tmp))

    return x_train, y_train, x_test, y_test


def run_cv_NewNetwork(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = NewNetwork(ep=ep)
        network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'outputs/y_test_NewNetwork29_{fold}_hpc.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'outputs/predicted_classes_NewNetwork29_{fold}_hpc.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'outputs/test_results_NewNetwork29_{fold}_hpc.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))


run_cv_NewNetwork(5, 10)
