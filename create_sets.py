"""
this python file contains all the create_sets functions
the create_functions' task is to create the train, validation and test sets, if needed
reshape and normalize the data

create_sets_random accepts two inputs: all of the data in sets "x" and "y"
create_sets_random -> using train_test_split, 67% of all data: train, 33%: test, 20% of train: validate
                        + normalizes and reshapes the data

create_sets_extra accepts two inputs: all of the data in sets "x" and "y"
create_set_extra -> create sets manually, first 67% of all data: train, 33%: test, 20% of train: validate

create_sets_manually accepts six inputs:
train data and labels, validation data and labels, test data and labels
create_sets_manually -> normalizes and reshapes the data

create_sets accepts four inputs: train data and labels, test data and labels
create_sets -> using train_test_split, 80% of train data: train, 20%: validate
                        + normalizes and reshapes the data

random state
if None we get different train and test sets across different executions and the shuffling process is out of control,
if 0 we get the same test and train sets across different executions
"""

from sklearn import model_selection as ms
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical


def create_sets_random(x, y):
    for i in range(x.shape[0]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x[i])
        x[i] = scaler.transform(x[i])

    x_train, x_test, y_train, y_test = ms.train_test_split(x, y, train_size=0.67, random_state=0)

    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    test_label = y_test_one_hot

    x_train, x_valid, train_label, valid_label = ms.train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=0)  # train and validate

    return x_train, train_label, x_valid, valid_label, x_test, test_label, y_test


def create_sets_extra(x, y):
    for i in range(x.shape[0]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x[i])
        x[i] = scaler.transform(x[i])

    n_train = int(x.shape[0]*0.67*0.8)  # 675 5 subjects
    n_valid = int(x.shape[0]*0.67)-n_train  # 169 5 subjects
    n_test = x.shape[0]-n_train-n_valid  # 416 5 subjects

    x_train = x[:n_train, :, :]  # first part of data: train
    y_train = y[:n_train]

    x_valid = x[n_train:n_train+n_valid, :, :]  # second part: validate
    y_valid = y[n_train:n_train+n_valid]

    x_test = x[n_train+n_valid:, :, :]  # rest: test
    y_test = y[n_train+n_valid:]

    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)

    train_label = to_categorical(y_train)
    valid_label = to_categorical(y_valid)
    test_label = to_categorical(y_test)

    return x_train, train_label, x_valid, valid_label, x_test, test_label, y_test


def create_sets_manually(x_train, y_train, x_valid, y_valid, x_test, y_test):
    for i in range(x_train.shape[0]):  # normalizing the train data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_train[i])
        x_train[i] = scaler.transform(x_train[i])

    for i in range(x_valid.shape[0]):  # normalizing the train data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_valid[i])
        x_valid[i] = scaler.transform(x_valid[i])

    for i in range(x_test.shape[0]):  # normalizing the test data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_test[i])
        x_test[i] = scaler.transform(x_test[i])

    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1)  # reshaping the train data
    x_valid = x_valid.reshape(-1, x_valid.shape[1], x_valid.shape[2], 1)  # reshaping the validation data
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)  # reshaping the test data

    train_label = to_categorical(y_train)
    valid_label = to_categorical(y_valid)
    test_label = to_categorical(y_test)

    return x_train, train_label, x_valid, valid_label, x_test, test_label, y_test


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