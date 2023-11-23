from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

path = 'C:/Users/Acer/Documents/7.felev/TDK/CVOutputsTDK/'


def load_y_test(add_path):
    y_test = []
    with open(path+add_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            y_test.append(int(line.strip()))
    f.close()
    return y_test


def load_predicted_classes(add_path):
    predicted_classes = []
    with open(path+add_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            predicted_classes.append(int(line.strip()))
    f.close()
    return predicted_classes


def visualize(y_test, predicted_classes, network_name=None, fold=None):
    cm = pd.DataFrame(confusion_matrix(y_test, predicted_classes),
                      columns=['0\nleft fist', '1\nright fist', '2\nboth fists', '3\nboth feet'],
                      index=['left fist\n0     ', 'right fist\n1      ', 'both fists\n2      ', 'both feet\n3      '])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.2)
    sn.heatmap(cm, annot=True, cmap='crest', fmt="d")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

    if network_name is None:
        plt.show()
    else:
        fname = path + network_name + '/'
        if network_name == 'EEGNet':
            fname = fname + f'EEGNet_new_params_avg4/EEGNet_new_params_avg4_figures/EEGNet_new_params_avg4_cv_{fold}.jpg'
        else:
            fname = path + network_name + f'/{network_name}_figures/{network_name}_cv_{fold}.jpg'
        plt.savefig(fname)


def plot_accuracy_and_loss(history, networkname, save=False, fname=None):
    # Accuracy
    plt.figure(figsize=(9, 6))
    sn.set(font_scale=1.2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(networkname+' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if save:
        plt.savefig(fname+'_accuracy.jpg')
    else:
        plt.show()

    # Loss
    plt.figure(figsize=(9, 6))
    sn.set(font_scale=1.2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(networkname+' Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if save:
        plt.savefig(fname + '_loss.jpg')
    else:
        plt.show()

