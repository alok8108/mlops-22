from sklearn import datasets

from sklearn.model_selection import train_test_split

# we will be preprocessing the data first
def digit_prepro(dataset):
    
    no_of_samples = len(dataset.images)
    shaped_data = dataset.images.reshape((no_of_samples, -1))
    label = dataset.target
    return shaped_data, label

#lets define the test split criterion
def same_test_split():
    Prop_train = 0.6
    prop_test = 0.2
    prop_test = 0.2
    
#Now load the data from Iris dataset
    digits = datasets.load_digits()
    shaped_data, label = digit_prepro(digits)

    # train data with random state=101
    
    x_train_1, x_test_1, y_train_1, y_test1 = train_test_split(
        shaped_data, label, test_size=prop_test, shuffle=True, random_state = 101
    )
    
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        shaped_data, label, test_size=test_frac, shuffle=True, random_state = 101
    )
    
   # Now check whthere train data is same or not
    assert (x_train_1 == X_train2).all()
    assert (x_test_1 == X_test2).all()
    assert (y_train_1 == y_train2).all()
    assert (y_test1 == y_test2).all()
    


def different_test_split():
    Prop_train = 0.8
    prop_test = 0.1
    prop_test = 0.1
    
    digits = datasets.load_digits()
    shaped_data, label = digit_prepro(digits)

    # lets train data with random state = Null
    
    x_train_1, x_test_1, y_train_1, y_test_1= train_test_split(
        shaped_data, label, test_size=prop_test, shuffle=True
    )
    
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
        shaped_data, label, test_size=prop_test, shuffle=True
    )
    
    # Now check whthere train data is same or not
    assert (x_train_1 == x_train_2).all() == False
    assert (x_test_1 == x_test_2).all() == False
    assert (y_train_1 == y_train_2).all() == False
    assert (y_test_1== y_test_2).all() == False

  