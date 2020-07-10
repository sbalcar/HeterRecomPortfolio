#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from configuration.configuration import Configuration #class
from datasets.behaviours import Behaviours #class

import matplotlib.pyplot as plt

def visualizationDHontModel():
   print("Visualization D'Hont Model")
   os.chdir("..")

   batchID:str = "ml1mDiv90Ulinear0109R1"

   fileName:str = "portfModelTimeEvolution-DHontRoulette.txt"
   jobID:str = fileName[fileName.index("-")+1:fileName.index(".")]
   #print(jobID)

   file:str = Configuration.resultsDirectory + os.sep + batchID + os.sep + fileName

   f = open(file, "r")

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



   plt.plot(recomTheMostPopularModel, label="RecomTheMostPopular", linewidth=0.25)
   plt.plot(recomCBmeanModel, label="RecomCBmean", linewidth=0.5)
   plt.plot(recomCBwindow3Model, label="RecomCBwindow3", linewidth=0.5)
   plt.plot(recomW2vPosnegMaxModel, label="RecomW2vPosnegMax", linewidth=0.5)
   plt.plot(recomW2vPosnegWindow3Model, label="RecomW2vPosnegWindow3", linewidth=0.5)

   plt.xlabel('x - time')
   plt.ylabel('y - method value')
   plt.title(batchID + " - " + jobID)

   plt.legend()
   plt.show()



   fig, axs = plt.subplots(5)
   fig.suptitle(batchID + " - " + jobID)
   axs[0].plot(recomTheMostPopularModel[0:100], label="RecomTheMostPopular")
   axs[0].legend()
   axs[1].plot(recomCBmeanModel[0:100], label="RecomCBmean")
   axs[1].legend()
   axs[2].plot(recomCBwindow3Model[0:100], label="RecomCBwindow3")
   axs[2].legend()
   axs[3].plot(recomW2vPosnegMaxModel[0:100], label="RecomW2vPosnegMax")
   axs[3].legend()
   axs[4].plot(recomW2vPosnegWindow3Model[0:100], label="RecomW2vPosnegWindow3")
   axs[4].legend()

   plt.show()

   print(recomCBmeanModel[0:10])
   print(recomW2vPosnegMaxModel[0:10])

visualizationDHontModel()