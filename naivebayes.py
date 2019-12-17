import pandas as pd
import numpy as np
import csv
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# https://medium.com/datadriveninvestor/implementing-naive-bayes-for-sentiment-analysis-in-python-951fa8dcd928

# ### PER MÉS ENDAVANT, MIRAR MILLORES A L'HORA DE TALLAR ELS TWEETS, COSES COM TREURE ELS COIXINETS O TREURE LES
# LLETRES REPETIDES ROTLO LOOOOOOONG

"""
Coses del Jan

ESTRUCTURA DE LES DADES
tweetId	    tweetText	                    tweetDate	sentimentLabel
1	        is so sad for my apl friend 	4/3/2015	0
2	        i miss the new moon trail 	    6/10/2015	0
53	        boom boom pow 	                29/9/2015	1

print(df[df["tweetId"]==1]["tweetText"])
print(df.shape[0])
print(df["sentimentLabel"].unique())

for index, element in df.iterrows():
    print(element["tweetId"])
    print(element["tweetText"])




# Crec que és innecessari tot això
class NaiveBayes:
    def __init__(self, df, colDades="tweetText", colLabel="sentimentLabel"):
        self.nTweets = df.shape[0]
        # self.vocabulari = self.calcula_vocab(df[colDades].tolist())
        self.diccionari = {}
        self.init_diccionari(df, colDades, colLabel)

        # jo seguiria la pagina que he deixat al comentari de dalt, sembla bastant nice com a primera implementació

    def calcula_vocab(self, dades):
        vocabulari = set()  # np.array
        for element in dades:
            for paraula in str(element).split(" "):
                # if paraula not in vocabulari:
                #    vocabulari.append(paraula)
                vocabulari.add(paraula.lower())
        return vocabulari  # set vocabulari amb totes les paraules úniques que apareixen a la bd

    def init_diccionari(self, df, colDades, colLabel):
        for labels in df[colLabel].unique():
            self.diccionari[labels] = []

        for i, element in df.iterrows():
            self.diccionari[element[colLabel]].append(element[colDades].split())

"""

def accuracy(nTP, nTN, nFP, nFN):
    return float(nTP + nTN) / (nTP + nTN + nFP + nFN)


def recall(nTP, nFN):
    if nTP + nFN == 0:
        return 0
    return float(nTP) / (nTP + nFN)


def precision(nTP, nFP):
    if nTP + nFP == 0:
        return 0
    else:
        return float(nTP) / (nTP + nFP)


def readData(file, rows):
    df = pd.read_csv(file, sep=";", nrows=rows)
    df = df.drop(['tweetId', 'tweetDate'], axis=1)
    return df


def splitData(df, trainPercentage = 70):
    # Holdout val 30 train 70
    ntrain = (df.shape[0] * trainPercentage) // 100
    # nval = df.shape[0]-ntrain
    train = df.loc[:ntrain]
    val = df.loc[ntrain + 1:]
    return train, val


def getTagDictionaries(train):
    pTrain = train.loc[train['sentimentLabel'] == 1]
    nTrain = train.loc[train['sentimentLabel'] == 0]

    pDict = {}
    nDict = {}

    totalWords = 0

    #No poseu el .keys() al fer if word in pDict perque va tot lent
    for tweet in pTrain['tweetText']:
        for word in str(tweet).split():
            if word in pDict:
                pDict[word] += 1
            else:
                pDict[word] = 1
                totalWords += 1

    for tweet in nTrain['tweetText']:
        for word in str(tweet).split():
            if word not in pDict:
                totalWords += 1
            if word in nDict:
                nDict[word] += 1
            else:
                nDict[word] = 1


    return pDict, nDict, totalWords


def taulaExtraccio(pDict, nDict, nUniqueWords, smoothing=1):
    """
    Calcula la taula de probabilitats de pertanyer a una classe o un altre,
     es un diccionari on la clau es la paraula amb una tupla en que el primer
      valor es yes(positiu), i el segon es no(negatiu).
    :return: Diccionari
    """
    taula = {}
    totalPositius = len(pDict)
    totalNegatius = len(nDict)
    for paraula, valor in pDict.items():
        numPositiu = valor
        if paraula in nDict:
            numNegatiu = nDict[paraula]
            total = numPositiu + numNegatiu
            """
            crec que aixo esta malament perque s'ha de fer que les columnes sumin 1 no les files
            pPositiu = numPositiu / total
            pNegatiu = 1 - pPositiu
            """
            pPositiu = (numPositiu + smoothing) / nUniqueWords
            pNegatiu = (numNegatiu + smoothing) / nUniqueWords
            taula[paraula] = (pPositiu, pNegatiu)
        else:
            taula[paraula] = ((numPositiu + smoothing) / nUniqueWords, 0 + smoothing / nUniqueWords)

    for paraula, valor in nDict.items():
        if paraula not in pDict:
            numNegatiu = valor
            pNegatiu = (numNegatiu + smoothing) / nUniqueWords
            taula[paraula] = ((0 + smoothing) / nUniqueWords, pNegatiu)

    return taula, totalPositius, totalNegatius


def printTaulaDeManeraMesBonica(taula):
    print("¦------------------------------------------------------------¦")
    print("¦          Paraula          ¦          Probabilitats         ¦")
    print("¦------------------------------------------------------------¦")
    print("¦                           ¦    Positiu     ¦    Negatiu    ¦")
    print("¦                           ¦--------------------------------¦")
    for paraula, tupla in taula.items():
        print("¦", paraula.ljust(25), "¦", str(round(tupla[0], 10)).ljust(14),
              "¦", str(round(tupla[1], 10)).ljust(13), "¦")
    print("¦------------------------------------------------------------¦")


#Per cuan cap paraula no estigui en el conjunt d'entrenament
default = False
def predict(taulaEx, valorationSet, priorPositive, priorNegative, totalPositius, totalNegatius, nUniqueWords, smoothing = 1):
    """
    Troba de cada tweet si es positiu o negatiu
    :return: Llista que diu si cada un dels tweets es positiu = True o negatiu = False
    """
    prediccions = []
    for tweet in valorationSet['tweetText']:
        pPositive = priorPositive
        pNegative = priorNegative
        noCalculable = True
        for word in str(tweet).split():
            if word in taulaEx:
                #Faig aixo per controlar si no tenim cap paraula del tweet al diccionari
                noCalculable = False
                pPositive *= taulaEx[word][0]
                pNegative *= taulaEx[word][1]
            else:
                """
                clar aixo vol dir que dons totes les probabilitats que teniem se li ha de sumar 1 al denominador,
                una possible solucio seria guardar totes les probabilitats a una llista multiplicarles pel denominador,
                i dividirles pel denominador + 1 i al final fer la multiplicació per tots els elements de la llista.
                Si no menteneu que es probable dema ho provo. Fin Xapa.
                """
                pPositive *= (smoothing / (nUniqueWords + smoothing))
                pNegative *= (smoothing / (nUniqueWords + smoothing))
        if noCalculable:
            prediccions.append(3)
        else:
            if pPositive > pNegative:
                prediccions.append(True)
            else:
                prediccions.append(False)

    return prediccions


def randomizeDataset(file):
    import random
    fid = open(file, "r")
    li = fid.readlines()
    fid.close()
    print(li)

    random.shuffle(li)
    print(li)

    fid = open("shuffled_example.txt", "w")
    fid.writelines(li)
    fid.close()


def validation(predictions, val):
    """
       Calcula les variables per a les metriques
       :return: True positive, true negative, false positive, false negative
    """
    i = 0
    """
    for tweet in val['tweetText']:
        print("Tweet:", tweet, "|", "Predicció:", prediccions[i], "\n")
        i += 1
    """
    """
    # Això va tot lent comparat amb lu de dalt el iterrows es una basura
    for index, row in val.iterrows():
        print("Tweet:", row['tweetText'], "|", "Predicció:", predictions[i], "|", "Valor actual: ",
              row['sentimentLabel'], "\n")
        i += 1
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for sentiment in val['sentimentLabel']:
        if predictions[i] == 3:
            pass
        elif predictions[i] == sentiment:
            if sentiment == 1:
                TP += 1
            else:
                TN += 1
        else:
            if sentiment == 0:
                FP += 1
            else:
                FN += 1
        i += 1

    return TP, TN, FP, FN

def nTweetsTest():
    file = 'data/shuffled_example.csv'
    trainPercentageTest = [0.001, 0.01, 0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]# [i*10 for i in range(1,10)]

    print(trainPercentageTest)
    rows = 1578628
    df = readData(file,rows)
    smoothing = 0
    acc = []
    rec = []
    pre = []
    for tp1 in trainPercentageTest:
        print("Evaluating network with %s of training data\n" % tp1)
        train, val = splitData(df, tp1)
        pDict, nDict, nUniqueWords = getTagDictionaries(train)
        taula, totalPositius, totalNegatius = taulaExtraccio(pDict, nDict, nUniqueWords, smoothing)
        priorPositive = totalPositius / (totalPositius + totalNegatius)
        priorNegatives = 1 - priorPositive
        prediccions = predict(taula, val, priorPositive, priorNegatives, totalPositius, totalNegatius, nUniqueWords, smoothing)
        tp, tn, fp, fn = validation(prediccions, val)

        acc.append(accuracy(tp, tn, fp, fn))
        rec.append(recall(tp, fn))
        pre.append(precision(tp, fp))

    #plt.figure()
    plt.plot(trainPercentageTest, acc, label = 'Accuracy')
    plt.plot(trainPercentageTest, rec, label = 'Recall')
    plt.plot(trainPercentageTest, pre, label = 'Precision')
    plt.xlabel("Train Size")
    plt.ylabel("%")
    plt.legend()
    plt.show()

def DicSizeTest():
    file = 'data/shuffled_example.csv'
    rows = 1578628
    df = readData(file,rows)
    smoothing = 0
    acc = []
    rec = []
    pre = []

    dictSizes = [10,50,100,1000, 10000,50000 ,100000,500000, 1000000]

    for size in dictSizes:
        train, val = splitData(df, 80)
        pDict, nDict, nUniqueWords = getTagDictionaries(train)
        taula, totalPositius, totalNegatius = taulaExtraccio(pDict, nDict, nUniqueWords, smoothing)
        reducedDict = {k: v for k, v in sorted(taula.items(), reverse=True, key=lambda item: item[1])[:size]}
        priorPositive = totalPositius / (totalPositius + totalNegatius)
        priorNegatives = 1 - priorPositive
        prediccions = predict(reducedDict, val, priorPositive, priorNegatives, totalPositius, totalNegatius, nUniqueWords, smoothing)
        tp, tn, fp, fn = validation(prediccions, val)

        acc.append(accuracy(tp, tn, fp, fn))
        rec.append(recall(tp, fn))
        pre.append(precision(tp, fp))

    plt.figure()
    plt.plot(dictSizes, acc, label = 'Accuracy')
    plt.plot(dictSizes, rec, label = 'Recall')
    plt.plot(dictSizes, pre, label = 'Precision')
    plt.xlabel("Dictionary size")
    plt.ylabel("%")
    plt.legend()
    plt.show()

def mainExe():
    file = 'data/shuffled_example.csv'
    # randomizeDataset(file) Només l'utiltzem una vegada per poder fer un analisi mes estable he vist que el fitxer
    # estava molt ordenat
    rows = 1578628
    df = readData(file, rows)
    train, val = splitData(df, 0.0001)
    smoothing = 0
    # Fase de train
    pDict, nDict, nUniqueWords = getTagDictionaries(train)
    print("\nDiccionari paraules positiveTag")
    print(pDict)
    print("\nDiccionari paraules negativeTag")
    print(nDict)
    print("\n")

    taula, totalPositius, totalNegatius = taulaExtraccio(pDict, nDict, nUniqueWords, smoothing)
    printTaulaDeManeraMesBonica(taula)

    priorPositive = totalPositius / (totalPositius + totalNegatius)
    priorNegatives = 1 - priorPositive

    # Prediccions
    prediccions = predict(taula, val, priorPositive, priorNegatives, totalPositius, totalNegatius, nUniqueWords, smoothing)

    tp, tn, fp, fn = validation(prediccions, val)

    acc = accuracy(tp, tn, fp, fn)
    print("Accuracy: ", acc * 100, "%")
    rec = recall(tp, fn)
    print("Recall: ", rec * 100, "%")
    pre = precision(tp, fp)
    print("Precision: ", pre * 100, "%")


def main():
    # nTweetsTest()
    # mainExe()
    DicSizeTest()




if __name__ == '__main__':
    main()


#print(priorPositive, priorNegatives)
#print(nUniqueWords)

# TODO: randomitzar el dataset cuan acabem de implementar tot aixo deixemlo aixi per debug purposes.

# TODO: podriem eliminar les paraules mes llargues de certa longitud per treure coses com links
#  ex: httptumblrcomxwp1yxhi6.

# TODO: afegir un spell checker com pyspellchecker.
