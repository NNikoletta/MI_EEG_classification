import run_functions as run
from load_data import load_cross_validation
import run_cross_validation as cv

import time

# cv.run_cv_ShallowConvNet(5, 10)
# cv.run_cv_EEGNet(5, 10)
# cv.run_cv_DSCNN(5, 10)
# cv.run_cv_MBEEGCBAM(5, 10)

# -----------------------------------------------------------------------------------

# run.run_ShallowConvNet(4, 2, 10)
# run.run_EEGNet(4, 2, 10)
# run.run_EEGNetCBAM(4, 2, 10)
# run.run_TwoBranchEEGNet(4, 2, 5)
# run.run_ThreeBranchEEGNet(4, 2, 5)
# run.run_FourBranchEEGNet(4, 2, 5)
# run.run_MBEEGCBAM(4, 2, 10)
# run.run_DSCNN(4, 2, 5)
# run.run_ProposedNetworkEEGNet(4, 2, 10)
# run.run_ProposedNetworkLSTM(4, 2, 10)

# ----------------------------------------------------------------------------------
#
from evaluate_results import load_y_test, load_predicted_classes, visualize
#
y_test = load_y_test('ProposedNetwork/FiltersAttentionAfterConcatenate/y_test_ProposedNetwork_00_hpc.txt')
predicted_classes = load_predicted_classes('ProposedNetwork/FiltersAttentionAfterConcatenate/predicted_classes_ProposedNetwork_00_hpc.txt')
visualize(y_test, predicted_classes)

