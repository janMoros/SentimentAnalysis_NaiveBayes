import pandas as pd
import numpy as np
import csv
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

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

"""


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


def accuracy(nTP, nTN, nFP, nFN):
    return float(nTP + nTN) / (nTP + nTN + nFP + nFN)


def recall(nTP, nFN):
    if (nTP + nFN) == 0:
        return 0
    return float(nTP) / (nTP + nFN)


def precision(nTP, nFP):
    return float(nTP) / (nTP + nFP)


def readData(file, rows):
    df = pd.read_csv(file, sep=";", nrows=rows)
    df = df.drop(['tweetId', 'tweetDate'], axis=1)
    return df


def splitData(df):
    # Holdout val 30 train 70
    ntrain = (df.shape[0] * 70) // 100
    # nval = df.shape[0]-ntrain
    train = df.loc[:ntrain]
    val = df.loc[ntrain + 1:]
    return train, val


def getTagDictionaries(train):
    pTrain = train.loc[train['sentimentLabel'] == 1]
    nTrain = train.loc[train['sentimentLabel'] == 0]

    pDict = {}
    nDict = {}

    for tweet in pTrain['tweetText']:
        for word in str(tweet).split():
            if word in pDict.keys():
                pDict[word] += 1
            else:
                pDict[word] = 1

    for tweet in nTrain['tweetText']:
        for word in str(tweet).split():
            if word in nDict.keys():
                nDict[word] += 1
            else:
                nDict[word] = 1

    return pDict, nDict


def taulaExtraccio(pDict, nDict):
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
            pPositiu = numPositiu / totalPositius
            pNegatiu = numNegatiu / totalNegatius
            taula[paraula] = (pPositiu, pNegatiu)
        else:
            taula[paraula] = (numPositiu / totalPositius, 0)

    for paraula, valor in nDict.items():
        if paraula not in pDict:
            numNegatiu = valor
            pNegatiu = numNegatiu / totalNegatius
            taula[paraula] = (0, pNegatiu)

    return taula


def printTaulaDeManeraMesBonica(taula):
    print("¦------------------------------------------------------------¦")
    print("¦          Paraula          ¦          Probabilitats         ¦")
    print("¦------------------------------------------------------------¦")
    print("¦                           ¦    Positiu     ¦    Negatiu    ¦")
    print("¦                           ¦--------------------------------¦")
    for paraula, tupla in taula.items():
        print("¦", paraula.ljust(25), "¦", str(round(tupla[0], 4)).ljust(14),
              "¦", str(round(tupla[1], 10)).ljust(13), "¦")
    print("¦------------------------------------------------------------¦")


#Per cuan cap paraula no estigui en el conjunt d'entrenament
default = False
def predict(taulaEx, valorationSet):
    """
    Troba de cada tweet si es positiu o negatiu
    :return: Llista que diu si cada un dels tweets es positiu = True o negatiu = False
    """
    prediccions = []
    for tweet in valorationSet['tweetText']:
        pPositive = 0
        pNegative = 0
        for word in str(tweet).split():
            if word in taulaEx:
                #Faig aixo per controlar si no tenim cap paraula del tweet al diccionari
                if pPositive == 0:
                    pPositive = 1
                if pNegative == 0:
                    pNegative = 1
                pPositive *= taulaEx[word][0]
                pNegative *= taulaEx[word][1]
        if pPositive == 0 and pNegative == 0:
            prediccions.append(default)
        else:
            if pPositive > pNegative:
                prediccions.append(True)
            else:
                prediccions.append(False)

    return prediccions


def main():
    file = 'data/FinalStemmedSentimentAnalysisDataset.csv'
    rows = 1000000
    df = readData(file, rows)
    train, val = splitData(df)

    # Fase de train
    pDict, nDict = getTagDictionaries(train)
    print("\nDiccionari paraules positiveTag")
    print(pDict)
    print("\nDiccionari paraules negativeTag")
    print(nDict)
    print("\n")

    taula = taulaExtraccio(pDict, nDict)
    printTaulaDeManeraMesBonica(taula)

    # Prediccions
    prediccions = predict(taula, val)
    i = 0
    for tweet in val['tweetText']:
        print("Tweet:", tweet, "|", "Predicció:", prediccions[i], "\n")
        i += 1


    # TODO: randomitzar el dataset cuan acabem de implementar tot aixo deixemlo aixi per debug purposes.

    # TODO: podriem eliminar les paraules mes llargues de certa longitud per treure coses com links
    #  ex: httptumblrcomxwp1yxhi6.

    # TODO: afegir un spell checker com pyspellchecker.


if __name__ == '__main__':
    main()
