SEED = None

# DATA Handling with the dataset in
#   /chillan_saved_images_square_224_ALL_with_len/<class ex: LP>/<id of event>_<length of signal>.png

# functions
# [x] - (debug/viz) how many images are there longer than THR for each class
# [x] - select only signals longer than THR (not in Already selected) => LONGER_THAN_THR1 and REST_SHORT
# [x] - select BALANCED subset (aka max C items from each class) and also return REST_UNBAL
# [x] - select A% of data and also return REMAINING_100-A%
# [x] - get Y from X image names

# WE WANT:
# training on signals > THR1
# sampling in a balanced manner (one class has 50k while other 5k)

# testing on signals > THR2a, THR2b, THR2c, ...
# not balanced
# randomly selected so they fit into memory! / or per batch

# select LONGER_THAN_THR1 and REST_SHORT
# from it BALANCED subset and REST_UNBAL

# TRAIN_BAL = 70% of BALANCED and REMAINING_30%
#
# VAL0 = (memory% or batch) REST_UNBAL + REST_SHORT + REMAINING_30%

# ... VAL0 will have a lot of unbalanced data ...
# alternatively:
# VAL = REMAINING_30% (just on similar looking data, that is also long enough and also balanced)
# VAL = longer than THR2abc (VAL0)

from os import listdir
from os.path import isfile, join
import numpy as np
import random
from matplotlib import pyplot as plt
#from keras.preprocessing.image import load_img, img_to_array
#import keras

def get_paths_of_all_images_from_folders(folders):
    X_all_paths = []
    Y_all_labels = []
    for i, folder in enumerate(folders):
        tmp = folder.split("/")
        label = tmp[-2]
        onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
        print(label, " occurs ", len(onlyfiles))
        onlyfiles = sorted(onlyfiles)
        paths = [folder + file for file in onlyfiles]
        Y_all_labels += [label] * len(paths)
        X_all_paths += paths

    return X_all_paths, Y_all_labels

def label_image_name(image_name):
    tmp = image_name.split(".jpg")
    tmp = tmp[0].split("/")
    label = tmp[-2]
    return label

def get_data_sorted_by_labels(images, unique_labels):
    by_label = {}
    for label in unique_labels:
        by_label[label] = []

    for image in images:
        label = label_image_name(image)
        by_label[label].append(image)

    for label in unique_labels:
        print(label, "occurs for", len(by_label[label]))

    return by_label


def debug_occurances_in_set(images, unique_labels):
    occurances = {}
    for label in unique_labels:
        occurances[label] = 0

    for image in images:
        label = label_image_name(image)
        occurances[label] += 1

    for label in unique_labels:
        print(label, "occurs for", occurances[label])

    return occurances


def shuffle_two_lists_together(a,b, SEED=None):
    if SEED is not None:
        random.seed(SEED)

    sort_order = random.sample(range(len(a)), len(a)) #random.shuffle(range(0,len(a)))
    a_new = [a[i] for i in sort_order]
    b_new = [b[i] for i in sort_order]
    a_new = np.asarray(a_new)
    b_new = np.asarray(b_new)
    return a_new, b_new

"""
THR1 = 1700
_,_ = how_many_images_longer_than_by_classes(THR1, images=X_all_paths, unique_labels=unique_labels)
"""


def sample_random_subset_from_list(L, N):
    # SHUFFLES!
    if len(L) < N:
        #print("less than N=",N,"data, selecting it all (without shuffle)")
        return L, []
    # warn this works inplace!
    random.shuffle(L)

    S = L[0:N]
    R = L[N+1:]
    #print("subset", S)
    return S, R

def select_balanced_set_max_C_from_one_class(C, images, unique_labels, v=True):
    # SHUFFLES!
    BALANCED = []
    REST = []

    images_by_categories = {}
    for label in unique_labels:
        images_by_categories[label] = []

    for image in images:
        label = label_image_name(image)
        images_by_categories[label].append(image)

    if v:
        for label in unique_labels:
            print("[Debug]",label,">>>")
            debug_occurances_in_set(images_by_categories[label], unique_labels)

    for label in unique_labels:
        set = images_by_categories[label]
        if v:
            if len(set) > C:
                print(len(set), "needs to be =< than", C)

        S, R = sample_random_subset_from_list(set, C)
        BALANCED += S
        REST += R

    #print("BALANCED", len(BALANCED), BALANCED)
    #print("REST", len(REST))

    return BALANCED, REST

"""
C = 5000
BALANCED, REST = select_balanced_set_max_C_from_one_class(C, images=X_all_paths, unique_labels=unique_labels)

debug_occurances_in_set(BALANCED,unique_labels)
debug_occurances_in_set(REST,unique_labels)
"""

def y_from_x(X):
    Y = []
    for i, path in enumerate(X):
        tmp = path.split("/")
        label = tmp[-2]
        Y.append(label)
    return Y

def split_data(x,y,validation_split=0.3, v=True):
    # Shuffle it before start!
    x, y = shuffle_two_lists_together(x, y, SEED=SEED)

    split_at = int(len(x) * (1 - validation_split))
    x_train = x[0:split_at]
    y_train = y[0:split_at]
    x_test = x[split_at:]
    y_test = y[split_at:]

    if v:
        print("Split", len(x), "images into", len(x_train), "train and", len(x_test), "test sets.")
    return x_train,y_train,x_test,y_test

"""
Y_all_labels = y_from_x(X_all_paths)
x_train, y_train, x_test, y_test = split_data(x=X_all_paths, y=Y_all_labels)

debug_occurances_in_set(x_train,unique_labels)
debug_occurances_in_set(x_test,unique_labels)
"""




# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------


# WE WANT:
# training on signals > THR1
# sampling in a balanced manner (one class has 50k while other 5k)

# testing on signals > THR2a, THR2b, THR2c, ...
# not balanced
# randomly selected so they fit into memory! / or per batch

# select LONGER_THAN_THR1 and REST_SHORT
# from it BALANCED subset and REST_UNBAL

# TRAIN_BAL = 70% of BALANCED and REMAINING_30%
#
# VAL0 = (memory% or batch) REST_UNBAL + REST_SHORT + REMAINING_30%

# ... VAL0 will have a lot of unbalanced data ...
# alternatively:
# VAL = REMAINING_30% (just on similar looking data, that is also long enough and also balanced)
# VAL = longer than THR2abc (VAL0)

# ------------------------------------------------------------------------------------------------------------------

def LOAD_DATASET_PREP(THR1, C_balanced, SPLIT, FOLDER, folders):
    X_all_paths, Y_all_labels = get_paths_of_all_images_from_folders(folders)
    X_all_paths, Y_all_labels = shuffle_two_lists_together(X_all_paths, Y_all_labels, SEED=SEED)

    unique_labels = np.unique(Y_all_labels)
    print("Unique labels:", unique_labels)

    LONGER_THAN_THR1 = X_all_paths
    REST_SHORT = []
    BALANCED, REST_UNBAL = select_balanced_set_max_C_from_one_class(C_balanced, LONGER_THAN_THR1, unique_labels, v=False)

    Y_BALANCED = y_from_x(BALANCED)
    X_TRAIN_BAL, _, X_REMAINING30, _ = split_data(x=BALANCED, y=Y_BALANCED, validation_split=SPLIT, v=False)
    return unique_labels, LONGER_THAN_THR1, REST_SHORT, BALANCED, REST_UNBAL, Y_BALANCED, X_TRAIN_BAL, X_REMAINING30

def LOAD_DATASET(THR1 = 1000, C_balanced = 5000, SPLIT = 0.3, FOLDER = 'chillan_saved_images_square_224_ALL_with_len',
    folders = ['data/chillan_saved_images_square_224_ALL_with_len/LP/', 'data/chillan_saved_images_square_224_ALL_with_len/TR/',
               'data/chillan_saved_images_square_224_ALL_with_len/VT/']):
    unique_labels, LONGER_THAN_THR1, REST_SHORT, BALANCED, REST_UNBAL, Y_BALANCED, X_TRAIN_BAL, X_REMAINING30 = \
        LOAD_DATASET_PREP(THR1, C_balanced, SPLIT, FOLDER, folders)

    print("")
    print("Training set:")
    debug_occurances_in_set(X_TRAIN_BAL, unique_labels)
    print("X names:", np.asarray(X_TRAIN_BAL).shape)

    REST_UNBAL = np.asarray(REST_UNBAL)
    REST_SHORT = np.asarray(REST_SHORT)
    X_REMAINING30 = np.asarray(X_REMAINING30)

    X_VAL_FULL = []

    #for s in REST_UNBAL:
    #    X_VAL_FULL.append(s)

    #for s in REST_SHORT:
    #    X_VAL_FULL.append(s)

    for s in X_REMAINING30:
        X_VAL_FULL.append(s)

    print("")
    print("Full Validation set:")
    debug_occurances_in_set(X_VAL_FULL, unique_labels)
    print(np.asarray(X_VAL_FULL).shape)
    print("X names:", np.asarray(X_VAL_FULL).shape)

    X_TRAIN_BAL = np.asarray(X_TRAIN_BAL)
    X_VAL_FULL = np.asarray(X_VAL_FULL)

    return X_TRAIN_BAL, X_VAL_FULL


def LOAD_DATASET_VAL_LONGER_THR2(THR1, C_balanced, SPLIT, FOLDER, folders, THR2, BalancedVal=True, StillBalance10to1to1 = True, C_balanced_2 = 10000):

    unique_labels, LONGER_THAN_THR1, REST_SHORT, BALANCED, REST_UNBAL, Y_BALANCED, X_TRAIN_BAL, X_REMAINING30 = \
        LOAD_DATASET_PREP(THR1, C_balanced, SPLIT, FOLDER, folders)

    print("")
    print("Training set:")
    debug_occurances_in_set(X_TRAIN_BAL, unique_labels)
    print("X names:", np.asarray(X_TRAIN_BAL).shape)

    REST_UNBAL = np.asarray(REST_UNBAL)
    REST_SHORT = np.asarray(REST_SHORT)
    X_REMAINING30 = np.asarray(X_REMAINING30)

    X_VAL_FULL = []

    # Do we allow unbalanced dataset????????
    if not BalancedVal:
        for s in REST_UNBAL:
            X_VAL_FULL.append(s)

    for s in REST_SHORT:
        X_VAL_FULL.append(s)

    for s in X_REMAINING30:
        X_VAL_FULL.append(s)

    # SELECT ONLY THOSE LONG
    #X_VAL_FULL, TOO_SHORT_FOR_VAL = select_signals_longer_than(THR2, X_VAL_FULL, v=False)
    TOO_SHORT_FOR_VAL = []
    # STILL BALANCE 10 to 1
    if StillBalance10to1to1:
        #C_balanced_2 = 10000
        #C_balanced_2 = 5000
        X_VAL_FULL, REST_UNBAL = select_balanced_set_max_C_from_one_class(C_balanced_2, X_VAL_FULL, unique_labels, v=False)


    print("")
    print("Full Validation set:")
    debug_occurances_in_set(X_VAL_FULL, unique_labels)
    print(np.asarray(X_VAL_FULL).shape)
    print("X names:", np.asarray(X_VAL_FULL).shape)

    X_TRAIN_BAL = np.asarray(X_TRAIN_BAL)
    X_VAL_FULL = np.asarray(X_VAL_FULL)

    return X_TRAIN_BAL, X_VAL_FULL


"""
# SAMPLE USAGE
THR1 = 1000
C_balanced = 5000
SPLIT = 0.3 # 70% and 30%
FOLDER = 'chillan_saved_images_square_224_ALL_with_len'
folders = ['data/'+FOLDER+'/LP/', 'data/'+FOLDER+'/TR/', 'data/'+FOLDER+'/VT/']

#LOAD_DATASET(THR1, C_balanced, SPLIT, FOLDER, folders)
LOAD_DATASET()
"""

########################################################################################
from keras.preprocessing.image import load_img, img_to_array

# load_images_with_keras, convert_labels_to_int
# convert_back_from_categorical_data, how_many_are_in_each_category

def load_images_with_keras(img_paths, target_size=None):
    imgs_arr = [img_to_array(load_img(path, target_size=target_size)) for path in img_paths]
    imgs_arr = np.array(imgs_arr)
    return imgs_arr

def convert_labels_to_int(Y, classes_names, labels):
    # classes_names[i] => labels[i]
    new_Y = []
    for y in Y:
        i = classes_names.index(y)
        l = labels[i]
        new_Y.append(l)
        #print(y, "to", l)
    return new_Y

def chunks(l, k):
    ''' Chunk data from list l into k fjords. Not randomizing the order. '''
    # the first chunk may have more data (if len(l) is not divisible by k)
    # if len(l) < k then the last chunks will be empty
    a = np.array_split(np.array(l), k)
    b = []
    for i in a:
        b.append(i.tolist())
    return b

def how_many_are_in_each_category(Y):
    unique_categories = set(Y)
    data_by_category = {}
    for cat in unique_categories:
        data_by_category[cat] = []
        for j in range(0,len(Y)):
            if Y[j] == cat:
                data_by_category[cat].append(Y[j])
    for cat in unique_categories:
        print(cat, " occured ", len(data_by_category[cat]))

def convert_back_from_categorical_data(Y):
    # manually written, rewrite
    # turn list of values according to this
    # (1,0,0) => 0
    # (0,1,0) => 1
    # (0,0,1) => 2
    new_Y = []
    for y in Y:
        k = np.argmax(y)
        new_Y.append(k)
    return new_Y
