import time
import statistics
from files.load_data import load_cross_validation
from files.create_sets import create_sets
import files.networks as net
from files.evaluate_results import plot_accuracy_and_loss

def run_cv_MBEEGCBAM(n_folds=5, ep=10):
    x_train_cv, y_train_cv, x_test_cv, y_test_cv = load_cross_validation(n_folds)

    test_losses = []
    test_accuracies = []

    for i in range(n_folds):
        x_train, train_label, x_valid, valid_label, x_test, test_label, y_test = create_sets(x_train_cv[i], y_train_cv[i], x_test_cv[i], y_test_cv[i])
        network = net.MBEEGCBAM(ep=ep)
        history = network.train(x_train, train_label, x_valid, valid_label)
        test_loss, test_acc = network.evaluate(x_test, test_label)
        predicted_classes = network.predict(x_test)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fold = '{:0>2}'.format(i)

        with open(f'outputs/y_test_MBEEGCBAM_{fold}_hpc.txt', 'w') as f:
            for k in range(len(y_test)):
                f.write(str(y_test[k]) + '\n')
        f.close()

        with open(f'outputs/predicted_classes_MBEEGCBAM_{fold}_hpc.txt', 'w') as f1:
            for k in range(len(predicted_classes)):
                f1.write(str(predicted_classes[k]) + '\n')
        f1.close()

        with open(f'outputs/test_results_MBEEGCBAM_{fold}_hpc.txt', 'w') as f2:
            f2.write("Test loss: " + str(test_loss) + " Test acc: " + str(test_acc) + '\n')
        f2.close()

        fname = f'outputs/MBEEGCBAM_{fold}_hpc'
        plot_accuracy_and_loss(history, 'MBEEGCBAM', True, fname)

    print("Mean loss:", statistics.mean(test_losses))
    print("Mean accuracy:", statistics.mean(test_accuracies))


start_time = time.time()
start_time_cpu = time.process_time()

run_cv_MBEEGCBAM(10, 10)

end_time = time.time()
end_time_cpu = time.process_time()

with open(f'outputs/runtime_MBEEGCBAM_hpc.txt', 'w') as f3:
    f3.write("Wall time: " + str(end_time-start_time) + " CPU time: " + str(end_time_cpu-start_time_cpu) + '\n')
f3.close()
