import sys
from visualize_history import visualize_history

print(len(sys.argv), " arguments : " , str(sys.argv))
FOLDER = '/home/vitek/Downloads/DTD/dtd-r1.0.1/dtd/images/'

if len(sys.argv)>1:
    FOLDER = str(sys.argv[1])

# SETUP

img_size = None #(20,20)
img_size = (150,150)
epochs = 750
batch_size = 32
validation_split = 0.2

PLOTNAME = 'classifier5FINE_'+str(epochs)+'epochs_'+str(validation_split)+'splits_'+str(img_size[0])+'imgsizes'


RESCALE = 1. / 255 # put data from 0-255 into 0-1
#RESCALE = 1

# GET ALL DATA
# define the classes in here directly


folders = ["banded", "crosshatched", "grid", "matted", "potholed", "studded", "blotchy", "crystalline", "grooved", "meshed", "scaly", "swirly", "braided", "dotted", "honeycombed", "paisley", "smeared", "veined", "bubbly", "fibrous", "interlaced", "perforated", "spiralled", "waffled", "bumpy", "flecked", "knitted", "pitted", "sprinkled", "woven", "chequered", "freckled", "lacelike", "pleated", "stained", "wrinkled", "cobwebbed", "frilly", "lined", "polka-dotted", "stratified", "zigzagged", "cracked", "gauzy", "marbled", "porous", "striped"]
classes_names = []
for a in folders:
    classes_names.append(a)
labels_texts = classes_names
num_classes = len(classes_names)

for i in range(0,len(folders)):
    folders[i] = FOLDER + folders[i] + "/"

labels = range(0,num_classes)

SHUFFLE_SEED=None

SPLIT = 0.3
DROP=0.2

print(folders)
print("---")
print(labels_texts)

from data_handling import LOAD_DATASET, LOAD_DATASET_VAL_LONGER_THR2, sample_random_subset_from_list, y_from_x
from data_handling import load_images_with_keras,  convert_labels_to_int, convert_back_from_categorical_data, how_many_are_in_each_category


specialname = '__PatternTest'
TRAIN_WITH_LONGER_THAN = 9999999
TRAIN_C_balanced = 9999999
C_balanced_2 = 9999999
VAL_ONLY_LONGER_THR2 = 9999999
BalancedVal = False
StillBalance10to1to1 = False
X_TRAIN_BAL, X_VAL_FULL = LOAD_DATASET_VAL_LONGER_THR2(
    TRAIN_WITH_LONGER_THAN, TRAIN_C_balanced, SPLIT, FOLDER, folders, VAL_ONLY_LONGER_THR2,
    BalancedVal=BalancedVal,StillBalance10to1to1 = StillBalance10to1to1, C_balanced_2=C_balanced_2)

SUBSET_FOR_TRAIN = 9999999
SUBSET_FOR_VAL = 9999999
#SUBSET_FOR_TRAIN = 100
#SUBSET_FOR_VAL = 100

############ Whats bellow doesn't have to be changed dramatically

X_TRAIN_BAL,_ = sample_random_subset_from_list(X_TRAIN_BAL, SUBSET_FOR_TRAIN)
Y_TRAIN_BAL = y_from_x(X_TRAIN_BAL)
X_VAL,_ = sample_random_subset_from_list(X_VAL_FULL, SUBSET_FOR_VAL)
Y_VAL = y_from_x(X_VAL)


from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import keras
from matplotlib import pyplot as plt

print("Loading image data!")
# X_TRAIN_BAL, Y_TRAIN_BAL
x_train = load_images_with_keras(X_TRAIN_BAL, target_size=img_size)
y_train = convert_labels_to_int(Y_TRAIN_BAL, classes_names, labels)
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)

# X_VAL, Y_VAL
x_test = load_images_with_keras(X_VAL, target_size=img_size)
y_test = convert_labels_to_int(Y_VAL, classes_names, labels)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)#, y_train[0:10])
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)#, y_test[0:10])
print("---")

print("SanityCheck Test dist:")
how_many_are_in_each_category(convert_back_from_categorical_data(y_test))
print("SanityCheck Train dist:")
how_many_are_in_each_category(convert_back_from_categorical_data(y_train))
print("---")

x_train *= RESCALE
x_test *= RESCALE

# =============================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================


from keras.applications import VGG16
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

from keras import models
from keras import layers
from keras import optimizers

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(DROP))
model.add(layers.Dense(num_classes, activation='sigmoid'))


#model.add(layers.Dense(num_classes, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

# Compile the model
# TRY
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4, clipnorm=1.),metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-6),metrics=['accuracy'])

# ====================================================================================

# x_train,y_train,x_test,y_test
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1)

model.save(PLOTNAME+"model.h5")

visualize_history(history.history, show_also='acc', save=True, show=False, save_path=PLOTNAME+specialname)

# ==============================================================================

### Now analyze results:
from sklearn.metrics import classification_report, confusion_matrix

# x_train,y_train,x_test,y_test

pred = model.predict(x_test, batch_size=32, verbose=1)
#y_predicted = np.argmax(pred, axis=1)
y_predicted = convert_back_from_categorical_data(pred)
#y_test_label = np.argmax(y_test, axis=1)
y_test_label = convert_back_from_categorical_data(y_test)
# Report
print("-------------------------------------------------------------------")
report = classification_report(y_test_label, y_predicted, target_names=classes_names)
print(report)

for i in range(0,len(labels)):
    print(labels[i],"=",classes_names[i])

# Confusion Matrix
cm = confusion_matrix(y_test_label,y_predicted)
cm=np.asarray(cm)
cm=cm.astype(float)

print(cm)

PercentagePerClass = True
if PercentagePerClass:
    # divide each row of cm by sum of the items in it
    # we can * 100 to get %
    for i, row in enumerate(cm):
        s = sum(row)
        #print("sum was",s)
        for j,r in enumerate(row):
            #print("r", r)
            #print("divided", float(r)/float(s))
            cm[i][j] = float(r)/float(s) * 100.0
print(cm)


# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd

df_cm = pd.DataFrame(cm, range(len(labels)),
                  range(len(labels)))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, cmap="YlGnBu", fmt=".2f", xticklabels=classes_names, yticklabels=classes_names)
#plt.show()

plt.savefig(PLOTNAME+"LastConfMatrix.png")
plt.savefig(PLOTNAME+"LastConfMatrix"+'.pdf', format='pdf')

