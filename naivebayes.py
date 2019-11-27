import pandas as pd
import numpy as np
#https://medium.com/datadriveninvestor/implementing-naive-bayes-for-sentiment-analysis-in-python-951fa8dcd928

#### PER MÉS ENDAVANT, MIRAR MILLORES A L'HORA DE TALLAR ELS TWEETS, COSES COM TREURE ELS COIXINETS O TREURE LES LLETRES REPETIDES ROTLO LOOOOOOONG


class NaiveBayes():
    def __init__(self, df, colDades="tweetText", colLabel="sentimentLabel"):
        self.nTweets = df.shape[0]
        #self.vocabulari = self.calcula_vocab(df[colDades].tolist())
        self.diccionari = {}
        self.init_diccionari(self, colDades, colLabel)


        # jo seguiria la pagina que he deixat al comentari de dalt, sembla bastant nice com a primera implementació


    def calcula_vocab(self, dades):
        vocabulari = set()#np.array
        for element in dades:
            for paraula in str(element).split(" "):
                #if paraula not in vocabulari:
                #    vocabulari.append(paraula)
                vocabulari.add(paraula.lower())
        return vocabulari  #set vocabulari amb totes les paraules úniques que apareixen a la bd

    def init_diccionari(self, colDades, colLabel):
        for labels in df[colLabel].unique():
            self.diccionari[labels] = []

        for i, element in df.iterrows():
            self.diccionari[element[colLabel]].append(element[colDades])


df = pd.read_csv('data/FinalStemmedSentimentAnalysisDataset.csv', sep=";")
""" ESTRUCTURA DE LES DADES
tweetId	    tweetText	                    tweetDate	sentimentLabel
1	        is so sad for my apl friend 	4/3/2015	0
2	        i miss the new moon trail 	    6/10/2015	0
53	        boom boom pow 	                29/9/2015	1
"""

"""
print(df[df["tweetId"]==1]["tweetText"])
print(df.shape[0])
print(df["sentimentLabel"].unique())
"""
prova = NaiveBayes(df)
print(prova.diccionari)
"""

for index, element in df.iterrows():
    print(element["tweetId"])
    print(element["tweetText"])
    """