import sys
print(len(sys.argv), " arguments : " , str(sys.argv))
FOLDER = '/home/vitek/Downloads/DTD/dtd-r1.0.1/dtd/images/'

if len(sys.argv)>1:
    FOLDER = str(sys.argv[1])

# SETUP

img_size = None #(20,20)
img_size = (150,150)
epochs = 150
batch_size = 32

validation_split = 0.3

RESCALE = 1. / 255 # put data from 0-255 into 0-1
#RESCALE = 1

# GET ALL DATA
# define the classes in here directly


folders = ["banded", "crosshatched", "grid", "matted", "potholed", "studded", "blotchy", "crystalline", "grooved", "meshed", "scaly", "swirly", "braided", "dotted", "honeycombed", "paisley", "smeared", "veined", "bubbly", "fibrous", "interlaced", "perforated", "spiralled", "waffled", "bumpy", "flecked", "knitted", "pitted", "sprinkled", "woven", "chequered", "freckled", "lacelike", "pleated", "stained", "wrinkled", "cobwebbed", "frilly", "lined", "polka-dotted", "stratified", "zigzagged", "cracked", "gauzy", "marbled", "porous", "striped"]
classes_names = folders
num_classes = len(classes_names)

for i in range(0,len(folders)):
    folders[i] = FOLDER + folders[i] + "/"

labels_texts = classes_names
labels = range(0,num_classes)

SHUFFLE_SEED=None
SUBSET = None #200 # (optional)

print(folders)


############ Whats bellow doesn't have to be changed dramatically

X_all_paths = []
Y_all_labels = []

from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import random
import keras
from visualize_history import visualize_history
from matplotlib import pyplot as plt


def load_images_with_keras(img_paths, target_size=None):
    imgs_arr = [img_to_array(load_img(path, target_size=target_size)) for path in img_paths]
    imgs_arr = np.array(imgs_arr)
    return imgs_arr

for i,folder in enumerate(folders):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    onlyfiles = sorted(onlyfiles)
    label = labels[i]
    #print(len(onlyfiles), "loaded", labels_texts[i], labels[i])
    #print(folder+onlyfiles[0])
    #print(folder+onlyfiles[-1])
    paths = [folder+file for file in onlyfiles]
    Y_all_labels += [label]*len(paths)
    X_all_paths += paths

X_all_image_data = load_images_with_keras(X_all_paths, target_size=img_size)
Y_all_labels = np.array(Y_all_labels)

Y_all_labels = keras.utils.to_categorical(Y_all_labels, num_classes=num_classes)

print("X_all_image_data:", X_all_image_data.shape)
print("Y_all_labels:", Y_all_labels.shape)
print("---")

VIZ = False
if VIZ:
    images = range(0,9)
    for i in images:
        plt.subplot(330 + 1 + i)
        #plt.imshow(X_all_image_data[i])
        plt.imshow((X_all_image_data[i] * 255).astype(np.uint8))

    #Show the plot
    plt.show()


VIZ = False
if VIZ:
    plt.hist(Y_all_labels, alpha=0.5)
    plt.title('Number of examples from each class')
    plt.xticks(np.arange(len(classes_names)), classes_names)
    plt.ylabel('count')
    plt.show()

# NOW WE HAVE ALL THE DATA X AND THEIR LABELS Y IN X_all_image_data, Y_all_labels
def shuffle_two_lists_together(a,b, SEED=None):
    if SEED is not None:
        random.seed(SEED)

    sort_order = list(range(0,len(a)))
    random.shuffle(sort_order)

    a_new = [a[i] for i in sort_order]
    b_new = [b[i] for i in sort_order]
    a_new = np.asarray(a_new)
    b_new = np.asarray(b_new)
    return a_new, b_new


def split_data(x,y,validation_split=0.2):
    split_at = int(len(x) * (1 - validation_split))
    x_train = x[0:split_at]
    y_train = y[0:split_at]
    x_test = x[split_at:]
    y_test = y[split_at:]

    print("Split", len(x), "images into", len(x_train), "train and", len(x_test), "test sets.")
    return x_train,y_train,x_test,y_test


X_all_image_data,Y_all_labels = shuffle_two_lists_together(X_all_image_data,Y_all_labels,SEED=SHUFFLE_SEED)
x_train,y_train,x_test,y_test = split_data(X_all_image_data,Y_all_labels,validation_split=validation_split)

# (optional) SUBSET
if SUBSET is not None:
    x_train = x_train[0:SUBSET]
    y_train = y_train[0:SUBSET]
    x_test = x_test[0:SUBSET]
    y_test = y_test[0:SUBSET]

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)#, y_train[0:10])
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)#, y_test[0:10])
print("---")

VIZ = False
if VIZ:
    images = range(0,9)
    for i in images:
        plt.subplot(330 + 1 + i)
        #plt.imshow(x_train[i])
        plt.imshow((x_train[i] * 255).astype(np.uint8))
    #Show the plot
    plt.show()

# See how is the distribution in test and train (should be the same)
VIZ = False
if VIZ:
    from matplotlib import pyplot as plt
    plt.subplot(2, 1, 1)
    plt.hist(y_train, alpha=0.5)
    plt.title('Train data classes')
    plt.xticks(np.arange(len(classes_names)), classes_names)
    plt.ylabel('count')

    plt.subplot(2, 1, 2)
    plt.hist(y_test, alpha=0.5)
    plt.title('Test data classes')
    plt.xticks(np.arange(len(classes_names)), classes_names)
    plt.ylabel('count')
    plt.show()

# Now for the model

"""
# LOAD AND TRAIN FULL MODEL
from keras import applications, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input

# LOAD VGG16
input_tensor = Input(shape=(img_size[0],img_size[1],3))
model = applications.VGG16(weights='imagenet',
                           include_top=False,
                           input_tensor=input_tensor)


# CREATE A TOP MODEL
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4, activation='sigmoid'))


# CREATE AN "REAL" MODEL FROM VGG16
# BY COPYING ALL THE LAYERS OF VGG16
new_model = Sequential()
for l in model.layers:
    new_model.add(l)

# CONCATENATE THE TWO MODELS
new_model.add(top_model)

# LOCK THE TOP CONV LAYERS
for layer in new_model.layers[:15]:
#for layer in new_model.layers:
    layer.trainable = False

# COMPILE THE MODEL
new_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', #optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
new_model.summary()
model = new_model

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=RESCALE)

# Show what we want to train on?

VIZ=False
if VIZ:
    img_rows, img_cols = img_size
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
        print(np.asarray(x_batch[0]).shape)
        print(x_batch[0].shape)

        # Show the first 9 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(x_batch[i].reshape(img_rows, img_cols, 3))
        # show the plot
        plt.show()
        break
if VIZ:
    img_rows, img_cols = img_size
    for x_batch, y_batch in datagen.flow(x_test, y_test, batch_size=9):
        print(np.asarray(x_batch[0]).shape)
        print(x_batch[0].shape)

        # Show the first 9 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(x_batch[i].reshape(img_rows, img_cols, 3))
        # show the plot
        plt.show()
        break

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=datagen.flow(x_test, y_test, batch_size=batch_size),
                        #validation_data=(x_test, y_test),
                        verbose=1)

"""


# WITH SAVED FEATURES - much faster
# but I had some doubts about the augmentation on/off on train and test data

from keras import applications
model = applications.VGG16(include_top=False, weights='imagenet')
from keras.utils import plot_model
plot_model(model, to_file='model_vgg.png', show_shapes=True)

# HERE WE ACTUALLY HAVE TO EDIT THE DATA OURSELF,
# aka x *= RESCALE

# predict(self, x, batch_size=None, verbose=0, steps=None)
x_train *= RESCALE
x_test *= RESCALE

X_bottleneck_train = model.predict(x_train, verbose=1)
X_bottleneck_test = model.predict(x_test, verbose=1)

print("X_bottleneck_train:", X_bottleneck_train.shape)
print("y_test:", y_train.shape)#, y_train[0:10])
print("X_bottleneck_test:", X_bottleneck_test.shape)
print("y_test:", y_test.shape)#, y_test[0:10])
print("---")

print("train_data.shape[1:]", X_bottleneck_train.shape[1:])

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers

model = Sequential()
model.add(Flatten(input_shape=X_bottleneck_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

plot_model(model, to_file='model_top.png', show_shapes=True)

# low LR
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-6),metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_bottleneck_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_bottleneck_test, y_test),
                    verbose=1)

visualize_history(history.history, show=False, show_also='acc', save=True, save_path='classifier3_'+str(epochs)+'epochs_')



# ==============================================================================



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


### Now analyze results:
from sklearn.metrics import classification_report, confusion_matrix

pred = model.predict(X_bottleneck_test, batch_size=32, verbose=1)
#y_predicted = np.argmax(pred, axis=1)
y_predicted = convert_back_from_categorical_data(pred)
#y_test_label = np.argmax(y_test, axis=1)
y_test_label = convert_back_from_categorical_data(y_test)
# Report
print("-------------------------------------------------------------------")
report = classification_report(y_test_label, y_predicted)
print(report)

for i in range(0,len(classes_names)):
    print(labels[i],"=",classes_names[i])

print("Val dataset:")
how_many_are_in_each_category(y_test_label)
print("Train dataset had:")
how_many_are_in_each_category(convert_back_from_categorical_data(y_train))

# Confusion Matrix
cm = confusion_matrix(y_test_label,y_predicted)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd

#df_cm = pd.DataFrame(cm, range(len(classes_names)-1),range(len(classes_names)-1))
df_cm = pd.DataFrame(cm, range(len(classes_names)),range(len(classes_names)))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, cmap="YlGnBu")

plt.savefig("LastConfMatrix")
plt.savefig("LastConfMatrix"+'.pdf', format='pdf')

#plt.show()
