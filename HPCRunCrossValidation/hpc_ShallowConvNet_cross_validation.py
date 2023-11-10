import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from sklearn import model_selection as ms
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Flatten
from sklearn.model_selection import KFold
import statistics
import time


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


def run_cv_ShallowConvNet(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = ShallowConvNet(ep=ep)
        network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'outputs/y_test_ShallowConvNet_{fold}_hpc.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'outputs/predicted_classes_ShallowConvNet_{fold}_hpc.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'outputs/test_results_ShallowConvNet_{fold}_hpc.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))


start_time = time.time()
start_time_cpu = time.process_time()

run_cv_ShallowConvNet(5, 10)

end_time = time.time()
end_time_cpu = time.process_time()

with open(f'outputs/runtime_ShallowConvNet_hpc.txt', 'w') as f3:
    f3.write("Wall time: " + str(end_time-start_time) + " CPU time: " + str(end_time_cpu-start_time_cpu) + '\n')
f3.close()
