import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import csv
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa  # pentru a putea printa F! score pe fiecae epoca fara a defini o functie custom pt callback
from keras.models import Sequential # Modelul pentru CNN
from keras.layers import Conv2D # Pentru un layer 2D de convolutie
from keras.layers import Dropout # Pentru aplicarea dropout-ului asupra inputului.
from keras.layers import MaxPool2D # Pentru aplicarea operatiei de max-pooling.
from keras.layers import Dense # Pentru un layer de NN conenctat dens.
from keras.layers import Flatten # Pentru aplatizarea inputului.
from keras.layers import MaxPooling2D
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np

folder_path = "unibuc-brain-ad/data/data/"

# dupa citirea fiecarei imagini, ii reduc dimensiunea la 128x128 
def modifyImage(img):     
        latime = int(128)
        inaltime = int(128)
        
        dimensiuni = (latime, inaltime)
        
        formatare = cv2.resize(img, dimensiuni, interpolation = cv2.INTER_AREA)
       
        return formatare

# incarc labelurile imaginilor de train
def loadTrainLabels():
        path = "unibuc-brain-ad/data/train_labels.txt"
        f = open(path, 'r')
        f.readline()

        trainLabels = []

        for line in f:
            id, label = line.split(',')
            labelFormatat = label[0]
            trainLabels.append(labelFormatat)
    
        return trainLabels  

# incarc labelurile imaginilor de validare
def loadValidationLabels():
        path = "unibuc-brain-ad/data/validation_labels.txt"
        f = open(path, 'r')
        f.readline()

        validationLabels = []

        for line in f:
            id, label = line.split(',')
            labelFormatat = label[0]
            validationLabels.append(labelFormatat)
        
        return validationLabels

# incarc imaginile de train si le salvez in 2 foldere diferite, in functie de label pentru
# a putea fi folosite ulterior pentru ImageDataGenerator
def load_images_from_folder_and_generate_new_Images( folder_path):
        trainLabelsInitial = loadTrainLabels()        

        i = 1
        # incarcam imaginile in intr-o lista pe care o impartim ulterior in functie de utilitate
        while i <= 15000:
            # salvez labelul imaginii pentru a vedea in ce folder o repartizez
            labelInInitialOrder = trainLabelsInitial[i-1]
            # citesc fiecare imagine
            img = cv2.imread(folder_path + "{:06d}".format(i) + ".png", cv2.IMREAD_GRAYSCALE)            
            # elimin noise-ul pt fiecare imagine
            img = cv2.fastNlMeansDenoising(img, None, 13, 4, 2)
            # modific dimensiunea imaginii
            img = modifyImage(img)
            # salvez fiecare imagine in folderul corespunzator
            if labelInInitialOrder == "1":
                cv2.imwrite("TrainImages/1/" + "{:06d}".format(i) + ".png", img)
            elif labelInInitialOrder == "0":
                cv2.imwrite("TrainImages/0/" + "{:06d}".format(i) + ".png", img)

            i+=1


# stocam imaginile de test si de validare
def loadValidationAndTest():
        # salvez la fel ca pt train labelurile pentru validare
        validationLabels = loadValidationLabels()
        i = 1

        while i <= 2000:
            labelInInitialOrder = validationLabels[i-1]
            # citesc fiecare imagine
            img = cv2.imread(folder_path + "{:06d}".format(i+15000) + ".png", cv2.IMREAD_GRAYSCALE)
            # modific dimensiunea imaginii
            img = modifyImage(img)

            # salvez fiecare imagine in folderul corespunzator
            if i <=2000:
                if labelInInitialOrder == "1":
                    cv2.imwrite("ValidationImages/Labeled1/" + "{:06d}".format(i+15000) + ".png", img)

                elif labelInInitialOrder == "0":
                    cv2.imwrite("ValidationImages/Labeled0/" + "{:06d}".format(i+15000) + ".png", img)
            i+=1

        # citesc imaginile pentru test
        i=1
        while i<=5149:
            img = cv2.imread(folder_path + "{:06d}".format(i+17000) + ".png", cv2.IMREAD_GRAYSCALE)
                
            img = modifyImage(img)

            cv2.imwrite("TestImages/" + "{:06d}".format(i+17000) + ".png", img)
            i+=1


#functia care va crea fisierul de output 
def createOutput(predictions):
        idx = 0
        output = "id,class\n"
        
        i = 17001
        while i < 22150:
            img = "{:06d},".format(i)
            row = img + str(int(predictions[idx][0])) + "\n"
            output += row
            idx += 1
            i+=1

        return output[:-1]

# functia pe care am testat-o sa analizez cum 
# evolueaza performantele modelului cu rata
# de invatare variabila 
def invscaling_lr(epoch, lr):
    if epoch == 0:
        return lr
    elif epoch % 7 == 0:
        return lr / (epoch ** 0.33)
    else:
        return lr

def create_cnnModel():
        # arhitectura modelului pe care am folosit-o si pe care o consider optima
        newCNN = Sequential()

        newCNN.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 1)))
        newCNN.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        newCNN.add(MaxPooling2D(pool_size=(2, 2)))
        newCNN.add(Dropout(0.25))

        newCNN.add(Conv2D(128, (3, 3), activation="relu"))
        newCNN.add(Conv2D(128, (3, 3), activation="relu"))
        newCNN.add(MaxPooling2D(pool_size=(2, 2)))
        newCNN.add(Dropout(0.25))

        newCNN.add(Conv2D(128, (3, 3), activation="relu"))
        newCNN.add(Conv2D(128, (3, 3), activation="relu"))
        newCNN.add(MaxPooling2D(pool_size=(2, 2)))
        newCNN.add(Dropout(0.25))

        newCNN.add(Flatten())

        newCNN.add(Dense(256, activation="relu"))
        newCNN.add(Dropout(0.5))

        newCNN.add(Dense(1, activation='sigmoid'))

        lr_scheduler = LearningRateScheduler(invscaling_lr)
        # pentru a putea folosi F1Score am instalat tensorflow_addons si am reusit sa printez F1 Score
        newCNN.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, average='micro', threshold=0.5)])
        newCNN.summary()
        return newCNN, lr_scheduler


# apelurile acestor doua functii le am comentate deoarece a fost indeajuns sa le rulez o singura data
# pentru a salva imaginile cum trebuie pentru ImageDataGenerator

# load_images_from_folder_and_generate_new_Images(folder_path=folder_path)
# loadValidationAndTest()

# incarc labelurile pentru train si validare
trainLabels = loadTrainLabels()
trainLabels = np.array(trainLabels).astype(np.int64)
validationLabels = loadValidationLabels()
validationLabels = np.array(validationLabels).astype(np.int64)


# definesc functia in care voi antrena modelul
def setPredictions():
        global trainLabels, validationLabels
        # initializez modelul
        modelCNN, lr= create_cnnModel()
        # modelCNN = load_model("epoci/cnn_80.h5")

        checkpoint_path = "epoci/cnn_{epoch:02d}.h5"
        # definesc acest checkpoint pentru a putea salva epocile cu scopul de a le analiza si de a
        # testa diferite comportamente ale modelului, astfel alegand o epoca optima
        checkpoint = ModelCheckpoint( 
            filepath=checkpoint_path,             
            save_weights_only=False,
            save_freq='epoch'
        )
        
        # definesc o serie de parametri pentru ImageDataGenerator care va genera noi imagini
        # augmentand astfel setul de date pentru train si vom avea mai multe imagini, deci
        # procesul de invatare va fi unul mai lent, dar mai eficient

        train_datagen = ImageDataGenerator(horizontal_flip = True, 
                                           rotation_range=20, 
                                           zoom_range=0.2, 
                                           shear_range=0.2, 
                                           rescale=1./255,
                                           width_shift_range=0.15,
                                           height_shift_range=0.15
                                        )
        
        # pentru fiecare imagine "initiala" vom genera noi imagini
        trainGenerator = train_datagen.flow_from_directory("TrainImages/", color_mode='grayscale', target_size = (128, 128), batch_size = 96, class_mode = 'binary')

        # procedam la fel pentru imaginile de validare, insa le vom da doar rescale
        validate_datagen = ImageDataGenerator(rescale=1./255)
        validationGenerator = validate_datagen.flow_from_directory("ValidationImages/", color_mode='grayscale', target_size = (128, 128), batch_size = 96, class_mode = 'binary')

        history = modelCNN.fit(trainGenerator, epochs = 300, validation_data=validationGenerator, callbacks=[checkpoint], class_weight={0:1, 1:1.5})
        

# functie care va citi imaginile de validare
def citesteValidateForPredict():
    i = 1
    imgValidationImagesForPredict = []

    while i <= 2000:
        img = cv2.imread("ValidationImagesForPredict/validation/" + "{:06d}".format(i+15000) + ".png", cv2.IMREAD_GRAYSCALE) 
        imgValidationImagesForPredict.append(img)
        i+=1

    imgValidationImagesForPredict = np.array(imgValidationImagesForPredict)
    return imgValidationImagesForPredict

# functie care va citi imaginile de test
def citesteTestForPredict():
    i = 1
    imgTestImagesForPredict = []

    while i <= 5149:
        img = cv2.imread("TestImages/test/" + "{:06d}".format(i+17000) + ".png", cv2.IMREAD_GRAYSCALE) 
        imgTestImagesForPredict.append(img)
        i+=1

    imgTestImagesForPredict = np.array(imgTestImagesForPredict)
    return imgTestImagesForPredict

def chooseBestEpoch():
        # normalizez imaginile pentru a se potrivi cu modul in care am lucrat cu cele de train
        testIMG_as_matrix = citesteTestForPredict() / 255.
        validationImages = citesteValidateForPredict() / 255.

        # reshape-uiesc imaginile 
        validationImages = validationImages.reshape(validationImages.shape[0], 128, 128, 1)
        testIMG_as_matrix = testIMG_as_matrix.reshape(testIMG_as_matrix.shape[0], 128, 128, 1)

        # aici incarc epoca pe care o consider optima in urma analizei tuturor epocilor
        modelCNN = load_model("epoci/cnn_256.h5")
        
        predictions = modelCNN.predict(testIMG_as_matrix)

        # pentru a putea oferi predictia va trebui sa aproximez rezultatul la 0 sau la 1
        predictionsForSubmission = [np.round(x) for x in predictions]
        output = createOutput(predictionsForSubmission)

        # pentru a analiza metricile, voi testa si pe imaginile de validare
        validationPrediction = modelCNN.predict(validationImages)
        validationPrediction = np.round(validationPrediction.reshape(len(validationPrediction))).astype(int)

        accuracy = accuracy_score(validationLabels, validationPrediction)
        f1_score_val = f1_score(validationLabels, validationPrediction)
        matrix = confusion_matrix(validationLabels, validationPrediction)
        from sklearn.metrics import classification_report
        classificationReport = classification_report(validationLabels, validationPrediction)
        print("Acuratete ", accuracy)
        print("F1-score ", f1_score_val)
        print("Matrice de confuzie:\n", matrix)
        print("classification report ", classificationReport)
        
        output = output.split("\n")
        idx = 0
        with open('submission.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in output:
                writer.writerow(row.split(","))

# apelam functia setPredictions() pentru a antrena modelul
setPredictions()

# apelam functia chooseBestEpoch() pentru a alege cea mai buna epoca
# (cea mai buna epoca se alege in functia chooseBestEpoch() setand parametrul pt functia load_model)
#  si pentru a oferi predictiile
chooseBestEpoch()