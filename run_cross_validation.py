from load_data import load_cross_validation
from create_sets import create_sets
import networks as net
import statistics


def run_cv_ShallowConvNet(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = net.ShallowConvNet(ep=ep)
        network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/ShallowConvNet/y_test_ShallowConvNet_{fold}_acer.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/ShallowConvNet/predicted_classes_ShallowConvNet_{fold}_acer.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/ShallowConvNet/test_results_ShallowConvNet_{fold}_acer.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))


def run_cv_EEGNet(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = net.EEGNet(ep=ep)
        network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/EEGNet/EEGNet_new_params_avg4/y_test_EEGNet_{fold}_acer.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/EEGNet/EEGNet_new_params_avg4/predicted_classes_EEGNet_{fold}_acer.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/EEGNet/EEGNet_new_params_avg4/test_results_EEGNet_{fold}_acer.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))


def run_cv_EEGNetCBAM(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = net.EEGNetCBAM(ep=ep)
        network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/EEGNetCBAM/y_test_EEGNetCBAM_{fold}_acer.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/EEGNetCBAM/predicted_classes_EEGNetCBAM_{fold}_acer.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/EEGNetCBAM/test_results_EEGNetCBAM_{fold}_acer.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))


def run_cv_DSCNN(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = net.DSCNN(ep=ep)
        network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/DSCNN/y_test_DSCNN_{fold}_acer.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/DSCNN/predicted_classes_DSCNN_{fold}_acer.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/DSCNN/test_results_DSCNN_{fold}_acer.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))


def run_cv_ProposedNetwork(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = net.ProposedNetwork(ep=ep)
        network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/ProposedNetwork/y_test_ProposedNetwork_{fold}_acer.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/ProposedNetwork/predicted_classes_ProposedNetwork_{fold}_acer.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/ProposedNetwork/test_results_ProposedNetwork_{fold}_acer.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))


def run_cv_NewNetwork(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = net.NewNetwork(ep=ep)
        network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/NewNetwork15/y_test_NewNetwork15_{fold}_acer.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/NewNetwork15/predicted_classes_NewNetwork15_{fold}_acer.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'C:/Users/Acer/Documents/7.felev/Thesis/CV_Outputs/NewNetwork15/test_results_NewNetwork15_{fold}_acer.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))
