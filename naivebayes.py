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
    primer valor es yes(positiu), segon es no(negatiu)
    """
    taula = []

    for paraula, valor in pDict.items():
        if paraula in nDict:
            numPositiu = valor
            numNegatiu = nDict[paraula]
            total = numPositiu + numNegatiu
            pPositiu = numPositiu / total
            pNegatiu = 1 - pPositiu
            taula.append([paraula, (pPositiu, pNegatiu)])
        else:
            taula.append([paraula, (1, 0)])

    for paraula in nDict.keys():
        if paraula not in pDict:
            taula.append([paraula, (0, 1)])

    return taula


def printTaulaDeManeraMesBonica(taula):
    print("¦------------------------------------------------------------¦")
    print("¦          Paraula          ¦          Probabilitats         ¦")
    print("¦------------------------------------------------------------¦")
    print("¦                           ¦    Positiu     ¦    Negatiu    ¦")
    print("¦                           ¦--------------------------------¦")
    for casella in taula:
        print("¦", casella[0].ljust(25), "¦", str(round(casella[1][0], 4)).ljust(14),
              "¦", str(round(casella[1][1], 4)).ljust(13), "¦")
    print("¦------------------------------------------------------------¦")


def main():
    file = 'data/FinalStemmedSentimentAnalysisDataset.csv'
    rows = 100
    df = readData(file, rows)
    train, val = splitData(df)
    pDict, nDict = getTagDictionaries(train)
    print("\nDiccionari paraules positiveTag")
    print(pDict)
    print("\nDiccionari paraules negativeTag")
    print(nDict)
    print("\n")
    taula = taulaExtraccio(pDict, nDict)
    printTaulaDeManeraMesBonica(taula)


if __name__ == '__main__':
    main()
