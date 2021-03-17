#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from configuration.configuration import Configuration #class
#from datasets.behaviours import Behaviours #class

import matplotlib.pyplot as plt

def visualizationDHontModel():
   print("Visualization Bandit Model")
   os.chdir("..")

   #batchID:str = "ml1mDiv90Ulinear0109R1"
   #batchID:str = "ml1mDiv90Ulinear0109R2"
   batchID: str = "online"

   #fileName:str = "portfModelTimeEvolution-BandiTS.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtThompsonSamplingINFFixedReduceProbOLin075025HLin05025.txt"

   fileName:str = "portfModelTimeEvolution-2.txt"
   #fileName:str = "portfModelTimeEvolution-4.txt"

   jobID:str = fileName[fileName.index("-")+1:fileName.index(".")]
   #print(jobID)

   inputFileName:str = Configuration.resultsDirectory + os.sep + batchID + os.sep + fileName
   inputFileName:str = "../resultsOnline" + os.sep + fileName

   f = open(inputFileName, "r")

   currentItemIDs:List[int] = []
   userIDs:List[int] = []
   methods:List[int] = []

   while True:
      currentItemIDLine:str = f.readline()
      if currentItemIDLine in ['']: break
      currentItemID:int = int(currentItemIDLine[currentItemIDLine.index(":") +1:-1])

      userIDLine:str = f.readline()
      userID:int = int(userIDLine[userIDLine.index(":") +1:-1])

      f.readline()
      f.readline()

      methodsI:dict = {}
      methodLineI:str = f.readline()
      while methodLineI not in ['\n', '\r\n']:
         print(methodLineI)
         wordsI:list[str] = methodLineI.split()
         print(wordsI)
         methodIDI = wordsI[0]
         print(wordsI[1])
         r = float(wordsI[1])
         n = float(wordsI[2])
         alpha0 = float(wordsI[3])
         beta0 = float(wordsI[4])

         methodsI[methodIDI] = n
         methodLineI = f.readline()

      currentItemIDs.append(currentItemID)
      userIDs.append(userID)
      methods.append(methodsI)


   # intialise data of lists.
   data:dict = {'currentItemID': currentItemIDs, 'userID': userIDs, 'methods':methods}
   # Create DataFrame
   df:DataFrame = DataFrame(data)
   print(df)

   recomThemostpopularModel:List[float] = []
   recomKnnModel:List[float] = []
   recomVmcontextknnModel:List[float] = []
   recomBpmmff50I20Lr01R003Model:List[float] = []
   recomBpmmff20I50Lr01R001Model:List[float] = []
   recomCoscbonemean1Model:List[float] = []
   recomCoscboneweightedmean5Model:List[float] = []
   recomW2Vtalli100000Ws1Vs32Upsmaxups1Model:List[float] = []
   recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Model:List[float] = []

   for indexI, rowI in df.iterrows():
      methodsI:dict = rowI['methods']
      #print(methodsI)
      recomThemostpopularModel.append(methodsI['RecomThemostpopular'])
      recomKnnModel.append(methodsI['RecomKnn'])
      recomVmcontextknnModel.append(methodsI['RecomVmcontextknn'])
      recomBpmmff50I20Lr01R003Model.append(methodsI['RecomBpmmff50I20Lr01R003'])
      recomBpmmff20I50Lr01R001Model.append(methodsI['RecomBpmmff20I50Lr01R001'])
      recomCoscbonemean1Model.append(methodsI['RecomCoscbonemean1'])
      recomCoscboneweightedmean5Model.append(methodsI['RecomCoscboneweightedmean5'])
      recomW2Vtalli100000Ws1Vs32Upsmaxups1Model.append(methodsI['RecomW2Vtalli100000Ws1Vs32Upsmaxups1'])
      recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Model.append(methodsI['RecomW2Vtalli200000Ws1Vs64Upsweightedmeanups5'])


   plt.plot(recomThemostpopularModel, label="Most pop.", linestyle='-', linewidth=0.7)
   plt.plot(recomKnnModel, label="iKNN", linestyle='--', linewidth=0.7)
   plt.plot(recomVmcontextknnModel, label="SKNN k:25", linestyle='--', linewidth=0.7)
   plt.plot(recomBpmmff50I20Lr01R003Model, label="BPR MF f:50, i:20, lr:0.1, reg:0.03", linestyle=':', linewidth=0.7)
   plt.plot(recomBpmmff20I50Lr01R001Model, label="BPR MF f:20, i:50, lr:0.1, reg:0.01", linestyle=':', linewidth=0.7)
   plt.plot(recomCoscbonemean1Model, label="Cosine CB agg:max, len:1", linestyle='-.', linewidth=0.7)
   plt.plot(recomCoscboneweightedmean5Model, label="Cosine CB agg:wAVG, len:5", linestyle='-.', linewidth=0.7)
   plt.plot(recomW2Vtalli100000Ws1Vs32Upsmaxups1Model, label="Word2vec f:32, w:1, i:100K, agg:max, len:1", linestyle='dashdot', linewidth=0.7)
   plt.plot(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Model, label="Word2vec f:64, w:1, i:200K, agg:wAVG, len:5", linestyle='dashdot', linewidth=0.7)


   plt.xlabel('x - time')
   plt.ylabel('y - method value')
   plt.title(batchID + " - " + jobID)

   plt.legend()
   #plt.show()

   outputFileName:str = ".." + os.sep + "images" + os.sep + batchID + fileName.replace(".txt", ".png")

   plt.savefig(outputFileName)



#   fig, axs = plt.subplots(5)
#   fig.suptitle(batchID + " - " + jobID)
#   axs[0].plot(recomTheMostPopularModel[0:100], label="RecomTheMostPopular")
#   axs[0].legend()
#   axs[1].plot(recomCBmeanModel[0:100], label="RecomCBmean")
#   axs[1].legend()
#   axs[2].plot(recomCBwindow3Model[0:100], label="RecomCBwindow3")
#   axs[2].legend()
#   axs[3].plot(recomW2vPosnegMaxModel[0:100], label="RecomW2vPosnegMax")
#   axs[3].legend()
#   axs[4].plot(recomW2vPosnegWindow3Model[0:100], label="RecomW2vPosnegWindow3")
#   axs[4].legend()
#
#   plt.show()
#
#   print(recomCBmeanModel[0:10])
#   print(recomW2vPosnegMaxModel[0:10])



if __name__ == "__main__":
   #os.chdir("..")

   visualizationDHontModel()