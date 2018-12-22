import numpy as np
import os
import shutil

def get_train_test(arr,validation_split):
    '''
    :param arr: list of data to be split
    :param validation_split: Split ratio, defaults to 80% for test set and 20% of validation set
    :return: Returns split shuffled and split data
    '''
    np.random.shuffle(arr)

    split_at = int(len(arr) * (1 - validation_split))
    arr_train = arr[0:split_at]
    arr_val = arr[split_at:]
    return arr_train,arr_val

def copy_files(files, path_to_files, target_folder):
    add = ""
    if path_to_files[-1:] is not "/":
        add = "/"
    for file in files:
        source = path_to_files+add+file
        destination = target_folder+file
        shutil.copyfile(source, destination)


def twofolders2fourfolders(firstclassfolder, secondclassfolder, train_test_split=0.3, enforce_same_number=False):
    '''
    processes two classes in two folders into training and testing subsets and accordingly into four folders

    :param firstclassfolder: folder containing images of the first class
    :param secondclassfolder: folder containing images of the second class
    :param train_test_split: 0.3 puts 30% to test and 70% to train
    :return:
    '''
    path_first = os.path.abspath(firstclassfolder)
    path_second = os.path.abspath(secondclassfolder)
    files_first = sorted(os.listdir(firstclassfolder))
    files_second = sorted(os.listdir(secondclassfolder))

    if enforce_same_number:
        np.random.shuffle(files_first)
        np.random.shuffle(files_second)

        min_i = np.min([len(files_first), len(files_second)])
        files_first = files_first[0:min_i]
        files_second = files_second[0:min_i]

    print("path_first:", path_first)

    train_files1, test_files1 = get_train_test(files_first, train_test_split)
    train_files2, test_files2 = get_train_test(files_second, train_test_split)

    print("first: train_files, test_files", len(train_files1), len(test_files1))
    print("second: train_files, test_files", len(train_files2), len(test_files2))
    #print("train_files", train_files1[0:10])
    #print("test_files", test_files1[0:10])

    # build file structure data/*train and validation*/*first and second*
    tmp = path_first.split("/")
    upfolder = path_first[0:-(len(tmp[-1]))]
    for folder in [upfolder + "data/train/first/", upfolder + "data/train/second/",
                   upfolder + "data/validation/first/", upfolder + "data/validation/second/"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # copy files
    folder = upfolder + "data/train/first/"
    copy_files(train_files1, path_first, folder)
    copy_files(test_files1, path_first, upfolder + "data/validation/first/")
    copy_files(train_files2, path_second, upfolder + "data/train/second/")
    copy_files(test_files2, path_second, upfolder + "data/validation/second/")

    return 0

if __name__ == '__main__':
    firstclassfolder = "no/"
    secondclassfolder = "yes/"
    train_test_split=0.3
    twofolders2fourfolders(firstclassfolder, secondclassfolder, train_test_split)