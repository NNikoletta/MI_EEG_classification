"""
this python file contains all the load functions
the load functions' only task is to download/load the recordings
subjects number 88, 89, 92, and 100 are excluded
load_random -> accepts one parameter: the number of subjects, load the data of the first n_subject subjects into x and y sets
load_by_sets -> accepts three parameters: the number of subjects used to create the train, the validation and test sets
load_train_and_test -> accepts two parameters: the number of subjects used to create the train and test sets
load_cross_validation -> accepts one parameter: the number of folds, loads the data of all subjects automatically
"""

import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import numpy as np
from sklearn.model_selection import KFold


def load_random(n_subjects):  # load the data of n_subjects number of subjects into x and y
    """
    The load_random function loads or downloads the EEG recordings from PhysioNet's website and concatenates all
    the data creating two datasets: x, and y.
    x contains all the trials of n_subjects in order of loading.
    y contains all the labels.
    :param n_subjects: number of subjects who's data is used (subjects 1-n_subjects)
    :return: x-data, y-labels
    """
    runs = [4, 6, 8, 10, 12, 14]  # use only hand and feet motor imagery runs
    epochs = []
    x = []
    y = []

    for subject in range(1, n_subjects+1):
        if subject != 88 and subject != 89 and subject != 92 and subject != 100:
            save_to = 'datasets'
            eegbci.load_data(subject, runs, save_to, verbose=0)  # downloading the files
            x_tmp = []
            y_tmp = []

            for record in runs:
                s = '{:0>3}'.format(subject)
                r = '{:0>2}'.format(record)
                filepath = f'datasets/MNE-eegbci-data/files/eegmmidb/1.0.0/S{s}/S{s}R{r}.edf'
                raw = read_raw_edf(filepath, verbose=0)  # getting raw data
                events, event_ids = mne.events_from_annotations(raw, verbose=0)  # events and their ids
                events = mne.pick_events(events, exclude=1)  # excluding rest, rest_id=1
                if record == 4 or record == 8 or record == 12:
                    events[:, 2] -= 2  # changing the ids: the first label is 0 instead of 2
                    event_dict = {
                        'left': 0,
                        'right': 1
                    }
                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9, verbose=0)
                    epoch.equalize_event_counts()
                    epochs.append(epoch)

                if record == 6 or record == 10 or record == 14:
                    event_dict = {
                        'fists': 2,
                        'feet': 3
                    }
                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9)
                    epoch.equalize_event_counts()
                    epochs.append(epoch)

                x_tmp.append(epoch.get_data())
                y_tmp.append(epoch.events[:, 2])
            x.append(np.concatenate(x_tmp))
            y.append(np.concatenate(y_tmp))

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y)
    return x, y


def load_by_sets(n_subjects_train, n_subjects_valid, n_subjects_test):
    runs = [4, 6, 8, 10, 12, 14]  # use only hand and feet motor imagery runs
    epochs = []
    x_train = []
    y_train = []

    x_valid = []
    y_valid = []

    x_test = []
    y_test = []

    for subject in range(1, n_subjects_train+n_subjects_test+n_subjects_valid+1):
        if subject != 88 and subject != 89 and subject != 92 and subject != 100:
            save_to = 'datasets'
            eegbci.load_data(subject, runs, save_to, verbose=0)  # downloading the files
            x_tmp_train = []
            y_tmp_train = []

            x_tmp_valid = []
            y_tmp_valid = []

            x_tmp_test = []
            y_tmp_test = []

            for record in runs:
                s = '{:0>3}'.format(subject)
                r = '{:0>2}'.format(record)
                filepath = f'datasets/MNE-eegbci-data/files/eegmmidb/1.0.0/S{s}/S{s}R{r}.edf'
                raw = read_raw_edf(filepath, verbose=0)  # getting raw data
                events, event_ids = mne.events_from_annotations(raw, verbose=0)  # events and their ids
                events = mne.pick_events(events, exclude=1)  # excluding rest, rest_id=1
                if record == 4 or record == 8 or record == 12:
                    events[:, 2] -= 2  # changing the ids: the first label is 0 instead of 2
                    event_dict = {
                        'left': 0,
                        'right': 1
                    }
                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9, verbose=0)
                    epoch.equalize_event_counts()
                    epochs.append(epoch)

                if record == 6 or record == 10 or record == 14:
                    event_dict = {
                        'fists': 2,
                        'feet': 3
                    }
                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9)
                    epoch.equalize_event_counts()
                    epochs.append(epoch)

                if subject <= n_subjects_train:
                    x_tmp_train.append(epoch.get_data())
                    y_tmp_train.append(epoch.events[:, 2])
                elif subject in range(n_subjects_train+1, n_subjects_train + n_subjects_valid + 1):
                    x_tmp_valid.append(epoch.get_data())
                    y_tmp_valid.append(epoch.events[:, 2])
                else:
                    x_tmp_test.append(epoch.get_data())
                    y_tmp_test.append(epoch.events[:, 2])
            if subject <= n_subjects_train:
                x_train.append(np.concatenate(x_tmp_train))
                y_train.append(np.concatenate(y_tmp_train))
            elif subject in range(n_subjects_train + 1, n_subjects_train + n_subjects_valid + 1):  # >= lower constraint < upper constraint
                x_valid.append(np.concatenate(x_tmp_valid))
                y_valid.append(np.concatenate(y_tmp_valid))
            else:
                x_test.append(np.concatenate(x_tmp_test))
                y_test.append(np.concatenate(y_tmp_test))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_train_and_test_old(n_subjects_train, n_subjects_test):
    runs = [4, 6, 8, 10, 12, 14]  # use only hand and feet motor imagery runs
    # epochs = []

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for subject in range(1, n_subjects_train+n_subjects_test+1):
        if subject != 88 and subject != 89 and subject != 92 and subject != 100:
            save_to = 'datasets'
            eegbci.load_data(subject, runs, save_to, verbose=0)  # downloading the files
            x_tmp_train = []
            y_tmp_train = []

            x_tmp_test = []
            y_tmp_test = []

            for record in runs:
                s = '{:0>3}'.format(subject)
                r = '{:0>2}'.format(record)
                filepath = f'datasets/MNE-eegbci-data/files/eegmmidb/1.0.0/S{s}/S{s}R{r}.edf'
                raw = read_raw_edf(filepath, verbose=0)  # getting raw data
                events, event_ids = mne.events_from_annotations(raw, verbose=0)  # events and their ids
                events = mne.pick_events(events, exclude=1)  # excluding rest, rest_id=1

                if record == 4 or record == 8 or record == 12:
                    events[:, 2] -= 2  # changing the ids: the first label is 0 instead of 2
                    event_dict = {
                        'left': 0,
                        'right': 1
                    }
                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9, verbose=0)
                    epoch.equalize_event_counts()
                    # epochs.append(epoch)

                if record == 6 or record == 10 or record == 14:
                    event_dict = {
                        'fists': 2,
                        'feet': 3
                    }
                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9, verbose=0)
                    epoch.equalize_event_counts()
                    # epochs.append(epoch)

                print(events, event_ids)

                if subject <= n_subjects_train:
                    x_tmp_train.append(epoch.get_data())
                    y_tmp_train.append(epoch.events[:, 2])
                else:
                    x_tmp_test.append(epoch.get_data())
                    y_tmp_test.append(epoch.events[:, 2])

            if subject <= n_subjects_train:
                x_train.append(np.concatenate(x_tmp_train))
                y_train.append(np.concatenate(y_tmp_train))
            else:
                x_test.append(np.concatenate(x_tmp_test))
                y_test.append(np.concatenate(y_tmp_test))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train)

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test)

    return x_train, y_train, x_test, y_test


def load_train_and_test(n_subjects_train, n_subjects_test):
    runs = [4, 6, 8, 10, 12, 14]  # use only hand and feet motor imagery runs
    # epochs = []

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for subject in range(1, n_subjects_train+n_subjects_test+1):
        if subject != 88 and subject != 89 and subject != 92 and subject != 100:
            save_to = 'datasets'
            eegbci.load_data(subject, runs, save_to, verbose=0)  # downloading the files
            x_tmp_train = []
            y_tmp_train = []

            x_tmp_test = []
            y_tmp_test = []

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


                if subject <= n_subjects_train:
                    x_tmp_train.append(epoch.get_data())
                    y_tmp_train.append(epoch.events[:, 2])
                else:
                    x_tmp_test.append(epoch.get_data())
                    y_tmp_test.append(epoch.events[:, 2])

            if subject <= n_subjects_train:
                x_train.append(np.concatenate(x_tmp_train))
                y_train.append(np.concatenate(y_tmp_train))
            else:
                x_test.append(np.concatenate(x_tmp_test))
                y_test.append(np.concatenate(y_tmp_test))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train)

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test)

    return x_train, y_train, x_test, y_test


def load_cross_validation_old(n_folds):
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
                events, event_ids = mne.events_from_annotations(raw, verbose=0)  # events and their ids
                events = mne.pick_events(events, exclude=1)  # excluding rest, rest_id=1

                if record == 4 or record == 8 or record == 12:
                    events[:, 2] -= 2  # changing the ids: the first label is 0 instead of 2
                    event_dict = {
                        'left': 0,
                        'right': 1
                    }
                    epoch = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.1, tmax=1.9, verbose=0)
                    epoch.equalize_event_counts()

                if record == 6 or record == 10 or record == 14:
                    event_dict = {
                        'fists': 2,
                        'feet': 3
                    }
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

    # for i, (train_index, test_index) in enumerate(kf.split(x)):
    #     print('Fold: ', i)
    #     print('Train', train_index)
    #     print('Test', test_index)

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
