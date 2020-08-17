from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from keras.layers import  MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
import random
import pathlib
import cv2
from constants import *

cough_dir = cough_path
non_cough_dir = non_cough_path

train_cough = [cough_dir + '{}'.format(i) for i in os.listdir(cough_dir) if 'p' in i]
train_non_cough = [non_cough_dir + '{}'.format(i) for i in os.listdir(non_cough_dir) if 'n' in i]
train_imgs = train_cough[:70] + train_non_cough[:70]
random.shuffle(train_imgs)


def read_proc_images(images_list):
    X = []
    y = []

    for file in images_list:
        print(file)
        X.append(cv2.resize(cv2.imread(file, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        if 'pos_chunks' in file:
            y.append(1)
        elif 'neg_chunks' in file:
            y.append(0)
    #for labels in y:
    #    print(labels)
    #print(len(y))
    #print(len(X))
    return X, y


def fifteen_sec_chunks(wav_file):
    # print(wav_file)
    count = 1
    for i in range(1, 1000, 15):
        t1 = i * 1000
        t2 = (i + 5) * 1000
        file_branch = wav_file.split('-')[0]
        file_id = wav_file.split('-')[1]
        file_stem = wav_file.split('-')[2]
        # print(file_branch)
        new_audio = AudioSegment.from_wav(wav_file)
        new_audio = new_audio[t1:t2]
        new_audio.export('{}_chunks/'.format(file_branch) + str(file_id) + '_' +
                         str(file_stem) + '_' + str(count) + '_fiftn' + '.wav', format="wav")
        count += 1


def ten_sec_chunks(wav_file):
    # print(wav_file)
    count = 1
    for i in range(1, 1000, 10):
        t1 = i * 1000
        t2 = (i + 5) * 1000
        file_branch = wav_file.split('-')[0]
        file_id = wav_file.split('-')[1]
        file_stem = wav_file.split('-')[2]
        # print(file_branch)
        new_audio = AudioSegment.from_wav(wav_file)
        new_audio = new_audio[t1:t2]
        new_audio.export('{}_chunks/'.format(file_branch) + str(file_id) + '_' +
                         str(file_stem) + '_' + str(count) + '_ten' + '.wav', format="wav")
        count += 1


def five_sec_chunks(wav_file):
    # print(wav_file)
    count = 1
    for i in range(1, 1000, 5):
        t1 = i * 1000
        t2 = (i + 5) * 1000
        file_branch = wav_file.split('-')[0]
        file_id = wav_file.split('-')[1]
        file_stem = wav_file.split('-')[2]
        # print(file_branch)
        new_audio = AudioSegment.from_wav(wav_file)
        new_audio = new_audio[t1:t2]
        new_audio.export('{}_chunks/'.format(file_branch) + str(file_id) + '_' +
                         str(file_stem) + '_' + str(count) + '_fve' + '.wav', format="wav")
        count += 1


def file_creator():
    mp3s = []
    for directories in paths:
        for objects in os.listdir(directories):
            mp3s.append(objects)

    for clips in mp3s:
        if 'neg' in clips:
            sound = AudioSegment.from_mp3(neg_path + clips)
            split_name = clips.split('.')[0]
            sound.export(split_name + ".wav", format="wav")
        else:
            sound = AudioSegment.from_mp3(pos_path + clips)
            split_name = clips.split('.')[0]
            sound.export(split_name + ".wav", format="wav")

    for objects in os.listdir(cwd):
        if '.wav' in objects:
            fifteen_sec_chunks(objects)
            ten_sec_chunks(objects)
            five_sec_chunks(objects)

    for g in labels:
        for files in os.listdir('{}/{}'.format(main_path, g)):
            if os.path.getsize('{}/{}/{}'.format(main_path, g, files)) < 100 * 1024:
                os.remove('{}/{}/{}'.format(main_path, g, files))

    for g in labels:
        type = g.replace('_', '')[0]
        # print(type)
        pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir('{}/{}'.format(main_path, g)):
            cough_split = '{}/{}/{}'.format(main_path, g, filename)
            y, sr = librosa.load(cough_split, mono=True, duration=5)
            # print(y.shape)
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, sides='default', mode='default', scale='dB')
            plt.axis('off')
            plt.savefig(f'img_data/{g}/{type + filename[:-3].replace(".", "")}.png')
            plt.clf()


def network_create():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='sigmoid', input_shape=(250, 250, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(12, (3, 3), activation='sigmoid'))
    model.add(MaxPooling2D((4, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='sigmoid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='sigmoid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.65))
    model.add(layers.Dense(48, activation='sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


def fit_keras_model(training_data, model):
    X, y = read_proc_images(training_data)
    for thing in X:
        print(thing)
    X = np.array(X)
    y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=2)

    ntrain = len(X_train)
    nval = len(X_val)

    print("Shape of train is:", X_train.shape)
    print("Shape of validation is:", X_val.shape)
    print("Shape of labels is :", y_train.shape)
    print("Shape of labels is:", y_val.shape)

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True, )

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    modelinfo = model.fit(train_generator,
                          steps_per_epoch=ntrain // batch_size,
                          epochs=32,
                          validation_data=val_generator,
                          validation_steps=nval // batch_size)
    return modelinfo


def validate_model(modelinfo):
    accuracy = modelinfo.history['acc']
    val_accuracy = modelinfo.history['val_acc']
    loss = modelinfo.history['loss']
    val_loss = modelinfo.history['val_loss']

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, 'b', label='Training accurarcy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()


def main():
    file_creator()
    model = network_create()
    results = fit_keras_model(train_imgs, model)
    validate_model(results)


if __name__ == "__main__":

    main()
