import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from skimage import transform
from sklearn.model_selection import train_test_split

# 1. set the ranges of hyper parameters 
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

assert len(h_param_comb) == len(gamma_list)*len(c_list)

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

print("\nImage size = ", {digits.images[0].shape})
#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
images_database = {}
def resize(size):
    print(f"\n----------FOR IMAGE SIZE ({size},{size})-----------\n")
    n_samples = len(digits.images)
    images_database[str(size)] = np.zeros((n_samples,size, size))
    for i in range(0,n_samples):
        images_database[str(size)][i] = transform.resize(digits.images[i],(size, size),anti_aliasing=True)
    data = images_database[str(size)].reshape((n_samples, -1))
    return data


    #PART: define train/dev/test splits of experiment protocol
    # train to train model
    # dev to set hyperparameters of the model
    # test to evaluate the performance of the model
def model_prediction(size = 8):
    data = resize(size)
    dev_test_frac = 1-train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
    )


    best_acc = -999
    best_h_params = None
    print("C, Gamma, Training Accuracy, Dev Accuracy, Test Accuracy\n")
    # 2. For every combination-of-hyper-parameter values
    for cur_parameters in h_param_comb:

        #PART: Define the model
        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        #PART: setting up hyperparameter
        hyper_params = cur_parameters
        clf.set_params(**hyper_params)


        #PART: Train model
        # 2.a train the model 
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # print(cur_parameters)
        #PART: get dev set predictions
        predict_train = clf.predict(X_train)
        predict_dev = clf.predict(X_dev)
        predict_test = clf.predict(X_test)
        # 2.b compute the accuracy on the validation set
        train_acc = metrics.accuracy_score(y_pred=predict_train, y_true=y_train)
        dev_acc = metrics.accuracy_score(y_pred=predict_dev, y_true=y_dev)
        test_acc = metrics.accuracy_score(y_pred=predict_test, y_true=y_test)
        print(cur_parameters['C'], cur_parameters['gamma'], train_acc, dev_acc, test_acc)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_parameters = cur_parameters

    print("Combinations of Hyperparameters found best:", end = " ")
    print(cur_parameters)

model_prediction()
model_prediction(12)
model_prediction(16)
model_prediction(20)
