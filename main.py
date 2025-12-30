import os
import pickle
import cv2
import numpy as np
import tkinter
from tkinter import *
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D 
from keras.models import Sequential, load_model 
from keras.callbacks import ModelCheckpoint 

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Initialize Main Window
main = tkinter.Tk()
main.title("Knee Osteoarthritis Stages Prediction")
main.geometry("1000x650")

# Global Variables
global filename
global classifier
global X, Y, X_train, y_train, X_test, y_test

# Updated unique labels
labels = ['Healthy', 'Mild', 'Moderate', 'Severe']

# Ensure model directory exists
if not os.path.exists("model"):
    os.makedirs("model")

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n\n')    
    text.insert(END, "Stages Found in Dataset : " + str(labels) + "\n\n")

def getID(name):
    for i in range(len(labels)):
        if labels[i].lower() in name.lower():
            return i
    return 0         

def preprocessDataset():
    text.delete('1.0', END)
    global filename, X, Y
    # Changed file extension to .npy for consistency
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root + "/" + directory[j])
                    if img is not None:
                        img = cv2.resize(img, (32, 32))
                        im2arr = np.array(img)
                        im2arr = im2arr.reshape(32, 32, 3)
                        X.append(im2arr)
                        label = getID(name)
                        Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.npy', X)
        np.save('model/Y.npy', Y)
        
    X = X.astype('float32') / 255
    text.insert(END, "Dataset Preprocessing Completed\n")
    text.insert(END, "Total images found in dataset : " + str(X.shape[0]) + "\n\n")
    text.update_idletasks()

def trainTest():
    text.delete('1.0', END)
    global X, Y, X_train, y_train, X_test, y_test
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y_cat = to_categorical(Y)
    
    # Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2)
    text.insert(END, "80% images used for Training : " + str(X_train.shape[0]) + "\n")
    text.insert(END, "20% images used for Testing  : " + str(X_test.shape[0]) + "\n")

def runCNN():
    text.delete('1.0', END)
    global X_train, y_train, X_test, y_test, classifier, labels
    
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(units=y_train.shape[1], activation='softmax'))
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    if not os.path.exists("model/model_weights.hdf5"):
        model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose=1, save_best_only=True)
        hist = classifier.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        with open('model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    else:
        classifier = load_model("model/model_weights.hdf5")
    
    predict_res = classifier.predict(X_test)
    predict_res = np.argmax(predict_res, axis=1)
    testY = np.argmax(y_test, axis=1)
    
    a = accuracy_score(testY, predict_res) * 100
    p = precision_score(testY, predict_res, average='macro') * 100
    r = recall_score(testY, predict_res, average='macro') * 100
    f = f1_score(testY, predict_res, average='macro') * 100
    
    text.insert(END, f"CNN Accuracy  : {a:.2f}\n")
    text.insert(END, f"CNN Precision : {p:.2f}\n")
    text.insert(END, f"CNN Recall    : {r:.2f}\n")
    text.insert(END, f"CNN F-Score   : {f:.2f}\n\n")
    
    conf_matrix = confusion_matrix(testY, predict_res)
    plt.figure(figsize=(6, 5)) 
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    plt.title("CNN Confusion Matrix") 
    plt.ylabel('True Class') 
    plt.xlabel('Predicted Class') 
    plt.show()

def graph():
    with open('model/history.pckl', 'rb') as f:
        data = pickle.load(f)
    accuracy = data['val_accuracy']
    error = data['val_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(accuracy, 'ro-', color='green', label='CNN Accuracy')
    plt.plot(error, 'ro-', color='red', label='CNN Loss')
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    plt.title('CNN Model Training Accuracy & Loss')
    plt.show()

def predict():
    global classifier, labels
    img_path = filedialog.askopenfilename(initialdir="testImages")
    if not img_path:
        return

    image = cv2.imread(img_path)
    img_resize = cv2.resize(image, (32, 32))
    img_array = np.array(img_resize).astype('float32') / 255
    img_array = img_array.reshape(1, 32, 32, 3)
    
    preds = classifier.predict(img_array)
    pred_idx = np.argmax(preds)
    prediction_label = labels[pred_idx]
    
    display_img = cv2.resize(image, (700, 400))
    cv2.putText(display_img, 'Predicted Stage: ' + prediction_label, 
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    
    cv2.imshow('Knee Osteoarthritis Prediction', display_img)
    cv2.waitKey(0)

def close():
    main.destroy()    

# GUI Layout
font_main = ('times', 15, 'bold')
title = Label(main, text='Knee Osteoarthritis Stages Prediction', bg='lavender blush', fg='DarkOrchid1', font=font_main, height=3, width=120)
title.pack()

font_btn = ('times', 12, 'bold')
Button(main, text="Upload Dataset", command=uploadDataset, font=font_btn).place(x=10, y=100)
Button(main, text="Preprocess Dataset", command=preprocessDataset, font=font_btn).place(x=200, y=100)
Button(main, text="Split Train/Test", command=trainTest, font=font_btn).place(x=400, y=100)
Button(main, text="Run CNN Algorithm", command=runCNN, font=font_btn).place(x=600, y=100)
Button(main, text="CNN Training Graph", command=graph, font=font_btn).place(x=10, y=150)
Button(main, text="Predict Stage", command=predict, font=font_btn).place(x=200, y=150)
Button(main, text="Exit", command=close, font=font_btn).place(x=400, y=150)

text = Text(main, height=20, width=120, font=('times', 12, 'bold'))
text.place(x=10, y=250)

main.config(bg='light coral')
main.mainloop()
