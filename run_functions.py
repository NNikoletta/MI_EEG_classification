from load_data import load_train_and_test
from create_sets import create_sets
import networks as net
from evaluate_results import visualize, plot_accuracy_and_loss


def run_ShallowConvNet(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.ShallowConvNet(ep=ep)
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'ShallowConvNet')


def run_EEGNet(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.EEGNet(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'EEGNet')


def run_TwoBranchEEGNet(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.TwoBranchEEGNet(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'Two-branch EEGNet')


def run_ThreeBranchEEGNet(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.ThreeBranchEEGNet(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'Three-branch EEGNet')


def run_FourBranchEEGNet(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.FourBranchEEGNet(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'Four-branch EEGNet')


def run_MBEEGCBAM(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.MBEEGCBAM(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'MBEEGCBAM')


def run_DSCNN(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.DSCNN(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'DSCNN')



def run_ProposedNetwork(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.ProposedNetwork(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'MultiConvNet')


def run_ProposedNetworkEEGNet(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.ProposedNetworkEEGNet(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'MultiConvNetEEG')


def run_ProposedNetworkLSTM(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.ProposedNetworkLSTM(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'MultiConvNetLSTM')
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def run_EEGNetCBAM(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.EEGNetCBAM(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'EEGNetCBAM')


def run_NewNetwork(n_subjects_train, n_subjects_test, ep=10):
    x_train, y_train, x_test, y_test = load_train_and_test(n_subjects_train, n_subjects_test)
    x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train, y_train, x_test, y_test)
    network = net.NewNetwork(ep=ep)  # change number of epochs here ep=...
    history = network.train(x_train, train_label, x_valid, valid_label)
    network.evaluate(x_test, test_label)
    predicted_classes = network.predict(x_test)
    visualize(y_test, predicted_classes)
    plot_accuracy_and_loss(history, 'NewNework')

