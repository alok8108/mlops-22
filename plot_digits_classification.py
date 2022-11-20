# """
# ================================
# Recognizing hand-written digits
# ================================

# This example shows how scikit-learn can be used to recognize images of
# hand-written digits, from 0-9.

# """

# # Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# # License: BSD 3 clause

# # Standard scientific Python imports
# import matplotlib.pyplot as plt

# # Import datasets, classifiers and performance metrics
# from sklearn import datasets, svm, metrics
# from sklearn.model_selection import train_test_split

# ###############################################################################
# # Digits dataset
# # --------------
# #
# # The digits dataset consists of 8x8
# # pixel images of digits. The ``images`` attribute of the dataset stores
# # 8x8 arrays of grayscale values for each image. We will use these arrays to
# # visualize the first 4 images. The ``target`` attribute of the dataset stores
# # the digit each image represents and this is included in the title of the 4
# # plots below.
# #
# # Note: if we were working from image files (e.g., 'png' files), we would load
# # them using :func:`matplotlib.pyplot.imread`.

# digits = datasets.load_digits()

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

# ###############################################################################
# # Classification
# # --------------
# #
# # To apply a classifier on this data, we need to flatten the images, turning
# # each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# # ``(64,)``. Subsequently, the entire dataset will be of shape
# # ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# # ``n_features`` is the total number of pixels in each image.
# #
# # We can then split the data into train and test subsets and fit a support
# # vector classifier on the train samples. The fitted classifier can
# # subsequently be used to predict the value of the digit for the samples
# # in the test subset.

# # flatten the images
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))

# # Create a classifier: a support vector classifier
# clf = svm.SVC(gamma=0.001)

# # Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False
# )

# # Learn the digits on the train subset
# clf.fit(X_train, y_train)

# # Predict the value of the digit on the test subset
# predicted = clf.predict(X_test)

# ###############################################################################
# # Below we visualize the first 4 test samples and show their predicted
# # digit value in the title.

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

# ###############################################################################
# # :func:`~sklearn.metrics.classification_report` builds a text report showing
# # the main classification metrics.

# print(
#     f"Classification report for classifier {clf}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# ###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# # disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# # disp.figure_.suptitle("Confusion Matrix")
# # print(f"Confusion matrix:\n{disp.confusion_matrix}")

# # plt.show()

from sklearn import datasets, svm, metrics, tree
from sklearn.metrics import accuracy_score, f1_score
import argparse
import pdb

from utils import (
    preprocess_digits,
    train_dev_test_split,
    data_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb(svm_params)

max_depth_list = [2, 10, 20, 50, 100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb(dec_params)

h_param_comb = {"svm": svm_h_param_comb, "decision_tree": dec_h_param_comb}

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

# define the evaluation metric
metric_list = [metrics.accuracy_score, macro_f1]
h_metric = metrics.accuracy_score

n_cv = 1
results = {}
# argparse
parser = argparse.ArgumentParser()


parser.add_argument('--clf_name', type=str, required=True)
# parser.add_argument('--tree', type=str, required=True)
parser.add_argument('--random_state', type=int, required=False)
args = parser.parse_args()

x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac, args.random_state
)

# for n in range(n_cv):
    # PART: Define the model
    # Create a classifier: a support vector classifier
models_of_choice = {
    "svm": svm.SVC(),
    # "decision_tree": tree.DecisionTreeClassifier(),
}
for clf_name in models_of_choice:
    clf = models_of_choice[clf_name]
    print("Running hyper param tuning for {}".format(clf_name))
    actual_model_path = tune_and_save(
        clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[clf_name], model_path=None
    )

    # 2. load the best_model
    best_model = load(actual_model_path)

    # PART: Get test set predictions
    # Predict the value of the digit on the test subset
    predicted = best_model.predict(x_test)
    # if not clf_name in results:
    #     results[clf_name]=[]    

    # results[clf_name].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
    # # 4. report the test set accurancy with that best model.
    # # PART: Compute evaluation metrics
    # print(
    #     f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n"
    # )



if args.clf_name == "svm":
    svm_acc = accuracy_score(y_test,predicted)
    svm_f1 = f1_score(y_test,predicted, average="macro")

    print("svm_acc:", svm_acc)
    print("svm_f1:", svm_f1)

elif args.clf_name == "tree":
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(x_train, y_train)


    decision_tree_p = decision_tree.predict(x_test)

    tree_acc = accuracy_score(y_test,decision_tree_p)
    tree_f1 = f1_score(y_test,predicted, average="macro")

    print("tree_acc:", tree_acc)
    print("tree_f1:", tree_f1)


print(results)