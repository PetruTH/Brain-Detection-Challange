import numpy as np
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import csv

folder_path = "unibuc-brain-ad/data/data/"
ct_labeled0 = 0

class RFC:
    def __init__(self):
        pass

    # functie utilizata pentru modificarea imaginii si a luminozitatii acesteia 
    def modifySizeAndBrightnessOfImage(self, img, procent = 50):
        procentScalare = procent 
        
        latime = int(img.shape[1] * procentScalare / 100)
        inaltime = int(img.shape[0] * procentScalare / 100)
        
        dimensiuni = (latime, inaltime)
        img = np.add(img, 10)
        
        formatare = cv2.resize(img, dimensiuni, interpolation = cv2.INTER_AREA)

        return formatare

    # obtin labelurile de antrenare
    def loadTrainLabels(self):
        path = "unibuc-brain-ad/data/train_labels.txt"
        f = open(path, 'r')
        f.readline()

        trainLabels = []

        for line in f:
            id, label = line.split(',')
            labelFormatat = label[0]
            trainLabels.append(labelFormatat)
    
        return trainLabels  

    # obtin labelurile pt validare
    def loadValidationLabels(self):
        path = "unibuc-brain-ad/data/validation_labels.txt"
        f = open(path, 'r')
        f.readline()

        validationLabels = []

        for line in f:
            id, label = line.split(',')
            labelFormatat = label[0]
            validationLabels.append(labelFormatat)
        
        return validationLabels

    # functie care citeste imaginile initiale
    def load_images_from_folder_and_generate_new_Images(self, folder_path):
        # declar path-ruile pentru fiecare tip de imagine (le voi salva ca np-arrays)
        storeTrainIMG = "unibuc-brain-ad/data/trainImages.npy"
        storeValidationIMG = "unibuc-brain-ad/data/validationImages.npy"
        storeTestIMG = "unibuc-brain-ad/data/testImages.npy"

        # pastrez pentru fiecare imagine de train labelul sau pentru a putea face UnderSampling 
        trainLabelsInitial = self.loadTrainLabels()

        # in aceste doua liste tin imaginile de train si labelurile lor
        # deoarece apelez la UnderSampling am nevoie de aceste doua structuri
        # pentru a pastra integritatea datelor si a nu amesteca labelurile 
        imagesResampledForTraining = []
        trainLabelsAfterUnderSampling = []

        # in aceste liste tin imaginile de validare si test
        validationImages = []
        testImages = []

        i = 1
        # citesc primele 15000 de imagini (cele de train)
        while i <= 15000:
            # pentru fiecare imagine ii extrag labelul
            labelInInitialOrder = trainLabelsInitial[i-1]
            
            # citesc imaginea
            img = cv2.imread(folder_path + "{:06d}".format(i) + ".png", cv2.IMREAD_GRAYSCALE)

            # modific dimensiunea si luminozitatea imaginii
            img = self.modifySizeAndBrightnessOfImage(img, procent=55)
            
            # transform imaginea intr-un np-array
            height, width = img.shape[:2]
            img = img.reshape(height * width)
            
            # UnderSampling pentru a balansa clasele
            global ct_labeled0

            # iau toate imaginile cu label 1
            if labelInInitialOrder == "1":
                imagesResampledForTraining.append(img)
                trainLabelsAfterUnderSampling.append(labelInInitialOrder)

            # iau doar 4500 de imagini cu label 0 
            elif labelInInitialOrder == "0" and ct_labeled0 < 4500:
                imagesResampledForTraining.append(img)
                trainLabelsAfterUnderSampling.append(labelInInitialOrder)

                ct_labeled0+=1


            i+=1

        # stocam imaginile de test si de validare
        i = 15001
        while i <= 22149:
            img = cv2.imread(folder_path + "{:06d}".format(i) + ".png", cv2.IMREAD_GRAYSCALE)
            
            img = self.modifySizeAndBrightnessOfImage(img, procent=55)
            img = np.ravel(img)

            if i <=17000:
                validationImages.append(img)
            elif i>17000:
                testImages.append(img)
            i+=1
        
        # le transform in arrayuri numpy pentru a putea fi folosite de algoritmul RFC
        trainImagesPixels = np.array(imagesResampledForTraining).astype(np.int64)
        validationImagesPixels = np.array(validationImages).astype(np.int64)
        testImagesPixels = np.array(testImages).astype(np.int64)

        # salvez array-urile in fisiere pentru a nu mai fi nevoie sa le incarcam de fiecare data
        np.save(storeTrainIMG, trainImagesPixels)
        np.save(storeValidationIMG, validationImagesPixels)
        np.save(storeTestIMG, testImagesPixels)

        # returnez noile labeluri pt train (care s-au schimbat in urma UnderSampling-ului)
        return trainLabelsAfterUnderSampling

    # functia care ne va ajuta sa construim fisierul submission.csv
    def createOutput(self, predictions, validation_OR_test):
        idx = 0
        output = "id,class\n"

        # daca validation_OR_test == 1 atunci vom crea fisierul pentru test
        # daca validation_OR_test == 0 atunci vom crea fisierul pentru validare
        if validation_OR_test == 1:
            i = 17001
            while i < 22150:
                img = "{:06d},".format(i)
                row = img + str(predictions[idx]) + "\n"
                output += row
                idx += 1
                i+=1
        elif validation_OR_test == 0:
            i = 15001
            while i < 17001:
                img = "{:06d},".format(i)
                row = img + str(predictions[idx]) + "\n"
                output += row
                idx += 1
                i+=1

        return output[:-1]

    # functia in care va avea loc antrenarea si testarea algoritmului RFC
    def setPredictions(self):
        # incarc datele dupa ce le-am prelucrat si am facut oversampling
        # Incarc datele pentru train
        trainLabels = self.load_images_from_folder_and_generate_new_Images(folder_path=folder_path)
        trainLabels = np.array(trainLabels).astype(np.int64)

        trainIMG_as_matrix = np.load("unibuc-brain-ad/data/trainImages.npy")

        # Incarc datele pentru test
        testIMG_as_matrix = np.load("unibuc-brain-ad/data/testImages.npy")


        # Incarc datele pentru validare
        validationLabels = self.loadValidationLabels()
        validationLabels = np.array(validationLabels).astype(np.int64)

        validationImages = np.load("unibuc-brain-ad/data/validationImages.npy")

        # definim modelul rfc cu hiperparametrii pe care ii consider optimi pentru acest task
        rfc = RandomForestClassifier(n_estimators=700, max_depth=70)

        # antrenez modelul
        rfc.fit(trainIMG_as_matrix, trainLabels)


        # ofer predictia pentru setul de test
        predictions = rfc.predict(validationImages)

        # ofer predictia pentru setul de validare si afisez si metricile pentru a
        # putea evalua complexitatea acestui model
        predictionsSubmit = rfc.predict(testIMG_as_matrix)
        print("Cccuracy " + str(accuracy_score(validationLabels, predictions)))
        print("Matrice de confuzie: \n", confusion_matrix(validationLabels, predictions))
        print("F1-score: " + str(f1_score(validationLabels, predictions, average='binary')))
        print("precision: " + str(classification_report(validationLabels, predictions)))

        # creez fisierul submission.csv pentru TEST sau pentru validation
        # apeland functia createOutput cu predictiile si cu un parametru
        # care ne spune daca este pentru test sau pentru validare
        output = self.createOutput(predictionsSubmit, validation_OR_test=1)

        output = output.split("\n")
        idx = 0
        with open('submission.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in output:
                writer.writerow(row.split(","))

if __name__ == "__main__":
    rfc = RFC()
    rfc.setPredictions()