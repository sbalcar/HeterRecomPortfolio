#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from configuration.configuration import Configuration #class
from datasets.behaviours import Behaviours #class

import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

def visualizationDHontModel():
   print("Visualization D'Hont Model")
   os.chdir("..")

   #batchID:str = "ml1mDiv90Ulinear0109R1"
   #batchID:str = "ml1mDiv90Ustatic08R1"
   batchID:str = "ml1mDiv90Ulinear0109R2"
   #batchID:str = "ml1mDiv90Ustatic08R2"

   #fileName:str = "portfModelTimeEvolution-DHontFixed.txt"
   #fileName:str = "portfModelTimeEvolution-DHontRoulette.txt"
   #fileName:str = "portfModelTimeEvolution-DHontRoulette3.txt"

   #fileName:str = "portfModelTimeEvolution-NegDHontFixedOLin0802HLin1002.txt"
   #fileName:str = "portfModelTimeEvolution-NegDHontFixedOStat08HLin1002.txt"
   #fileName:str = "portfModelTimeEvolution-NegDHontRouletteOLin0802HLin1002.txt"
   #fileName:str = "portfModelTimeEvolution-NegDHontRouletteOStat08HLin1002.txt"
   fileName:str = "portfModelTimeEvolution-NegDHontRoulette3OLin0802HLin1002.txt"
   #fileName:str = "portfModelTimeEvolution-NegDHontRoulette3OStat08HLin1002.txt" ##

   jobID:str = fileName[fileName.index("-")+1:fileName.index(".")]
   #print(jobID)

   inputFileName:str = Configuration.resultsDirectory + os.sep + batchID + os.sep + fileName

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
         methodIDI:str = methodLineI[0:methodLineI.index(" ")]
         votesI:float = float(methodLineI[methodLineI.index("0."):-1])
         methodsI[methodIDI] = votesI
         methodLineI = f.readline()

      currentItemIDs.append(currentItemID)
      userIDs.append(userID)
      methods.append(methodsI)


   # intialise data of lists.
   data:dict = {'currentItemID': currentItemIDs, 'userID': userIDs, 'methods':methods}
   # Create DataFrame
   df:DataFrame = DataFrame(data)
   print(df)

   recomTheMostPopularModel:List[float] = []
   recomCBmeanModel:List[float] = []
   recomCBwindow3Model:List[float] = []
   recomW2vPosnegMaxModel:List[float] = []
   recomW2vPosnegWindow3Model:List[float] = []

   for indexI, rowI in df.iterrows():
      methodsI:dict = rowI['methods']
      #print(methodsI)
      recomTheMostPopularModel.append(methodsI['RecomTheMostPopular'])
      recomCBmeanModel.append(methodsI['RecomCBmean'])
      recomCBwindow3Model.append(methodsI['RecomCBwindow3'])
      recomW2vPosnegMaxModel.append(methodsI['RecomW2vPosnegMax'])
      recomW2vPosnegWindow3Model.append(methodsI['RecomW2vPosnegWindow3'])

   recomTheMostPopularLabel:str = "Most Popular"
   recomCBmeanLabel:str = "Cosine CB; max"
   recomCBwindow3Label:str = "Cosine CB; last-3"
   recomW2vPosnegMaxLabel:str = "Word2vec; mean"
   recomW2vPosnegWindow3Label:str = "Word2vec; last-3"

   windowSize:int = 801
   polynomialOrder:int = 1
   recomTheMostPopularModelSF:List[float] = savgol_filter(recomTheMostPopularModel, windowSize, polynomialOrder)  # window size 51, polynomial order 3
   recomCBmeanModelSF:List[float] = savgol_filter(recomCBmeanModel, windowSize, polynomialOrder)
   recomCBwindow3ModelSF:List[float] = savgol_filter(recomCBwindow3Model, windowSize, polynomialOrder)
   recomW2vPosnegMaxModelSF:List[float] = savgol_filter(recomW2vPosnegMaxModel, windowSize, polynomialOrder)
   recomW2vPosnegWindow3ModelSF:List[float] = savgol_filter(recomW2vPosnegWindow3Model, windowSize, polynomialOrder)

   y:List[int] = range(107538)
   #plt.figure(figsize=(4, 7))
   plt.figure(figsize=(7, 3.5))

   plt.plot(recomTheMostPopularModel, linewidth=0.02)
   plt.plot(recomCBmeanModel, linewidth=0.02)
   plt.plot(recomCBwindow3Model, linewidth=0.02)
   plt.plot(recomW2vPosnegMaxModel, linewidth=0.02)
   plt.plot(recomW2vPosnegWindow3Model, linewidth=0.02)

   plt.gca().set_prop_cycle(None)

   plt.plot(recomTheMostPopularModelSF, label=recomTheMostPopularLabel, linewidth=1.0)
   plt.plot(recomCBmeanModelSF, label=recomCBmeanLabel, linewidth=1.0)
   plt.plot(recomCBwindow3ModelSF, label=recomCBwindow3Label, linewidth=1.0)
   plt.plot(recomW2vPosnegMaxModelSF, label=recomW2vPosnegMaxLabel, linewidth=1.0)
   plt.plot(recomW2vPosnegWindow3ModelSF, label=recomW2vPosnegWindow3Label, linewidth=1.0)

   #plt.xticks(rotation=45)
   #plt.yticks(rotation=90)

   #plt.ylabel('y - time', labelpad=-725)
   plt.xlabel('Votes assignment values')
   #plt.title(batchID + " - " + jobID)

   plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.)

   outputFileName:str = ".." + os.sep + "images" + os.sep + batchID + fileName.replace(".txt", ".png")

   #plt.savefig(outputFileName)
   plt.show()



   fig, axs = plt.subplots(5)
   fig.suptitle(batchID + " - " + jobID)
   axs[0].plot(recomTheMostPopularModel, label=recomTheMostPopularLabel)
   axs[0].legend()
   axs[1].plot(recomCBmeanModel, label=recomCBmeanLabel)
   axs[1].legend()
   axs[2].plot(recomCBwindow3Model, label=recomCBwindow3Label)
   axs[2].legend()
   axs[3].plot(recomW2vPosnegMaxModel, label=recomW2vPosnegMaxLabel)
   axs[3].legend()
   axs[4].plot(recomW2vPosnegWindow3Model, label=recomW2vPosnegWindow3Label)
   axs[4].legend()

   #plt.show()
   #plt.savefig("para" + batchID + fileName)

   print(recomCBmeanModel[0:10])
   print(recomW2vPosnegMaxModel[0:10])


visualizationDHontModel()