#!/usr/bin/python3

import os

from typing import List

from pandas.core.frame import DataFrame #class

from configuration.configuration import Configuration #class

import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

import seaborn as sns


def visualizationDHondtModelML():
   print("Visualization D'Hont Model")
   #os.chdir("..")

   #batchID:str = "ml1mDiv90Ulinear0109R1"
   #batchID:str = "ml1mDiv90Ulinear0109R2"
   #batchID:str = "ml1mDiv90Upowerlaw054min048R1"
   #batchID:str = "ml1mDiv90Upowerlaw054min048R2"
   #batchID:str = "ml1mDiv90Ustatic08R1"
   batchID:str = "ml1mDiv90Ustatic08R2"

   #fileName:str = "portfModelTimeEvolution-DHontFixed.txt"
   #fileName:str = "portfModelTimeEvolution-DHontRoulette.txt"
   #fileName:str = "portfModelTimeEvolution-DHontRoulette3.txt"

   fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin07500HLin07500.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin1005HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin10075HLin0500.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin07500HLin07500.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin075025HLin05025.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin1005HLin1005.txt"

   #fileName:str = "portfModelTimeEvolution-ContextDHondtINFFixedReduceOLin07500HLin07500.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtINFFixedReduceOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtINFFixedReduceOLin1005HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtINFFixedReduceOLin10075HLin0500.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtINFFixedReduceProbOLin07500HLin07500.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtINFFixedReduceProbOLin075025HLin05025.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtINFFixedReduceProbOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtINFFixedReduceProbOLin1005HLin1005.txt"

   #fileName:str = "portfModelTimeEvolution-ContextDHondtRoulette1.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeFixed.txt"

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
         #print(methodLineI)
         if "0." in methodLineI:
            votesI:float = float(methodLineI[methodLineI.index("0."):-1])
         elif "-" in methodLineI:
            votesI:float = float(methodLineI[methodLineI.index("-"):-1])

         methodsI[methodIDI] = votesI
         methodLineI = f.readline()

      currentItemIDs.append(currentItemID)
      userIDs.append(userID)
      methods.append(methodsI)


   # intialise data of lists.
   data:dict = {'currentItemID': currentItemIDs, 'userID': userIDs, 'methods':methods}
   # Create DataFrame
   df:DataFrame = DataFrame(data)
   #print(df)

   recomTheMostPopularModel:List[float] = []
   recomKnnModel:List[float] = []
   recomVmcontextknn25Model:List[float] = []
   recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Model:List[float] = []
   recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Model:List[float] = []
   recomCosinecbcbdoheupsweightedmeanups3Model:List[float] = []
   recomCosinecbcbdoheupsmaxups1Model:List[float] = []
   recomBprmff100I10Lr0003R01Model:List[float] = []
   recomBprmff20I20Lr0003R01Model:List[float] = []

   for indexI, rowI in df.iterrows():
      methodsI:dict = rowI['methods']
      #print(methodsI)
      recomTheMostPopularModel.append(methodsI['RecomThemostpopular'])
      recomKnnModel.append(methodsI['RecomKnn'])
      recomVmcontextknn25Model.append(methodsI['RecomVmcontextknn25'])

      recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Model.append(methodsI['RecomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3'])
      recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Model.append(methodsI['RecomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7'])
      recomCosinecbcbdoheupsweightedmeanups3Model.append(methodsI['RecomCosinecbcbdoheupsweightedmeanups3'])
      recomCosinecbcbdoheupsmaxups1Model.append(methodsI['RecomCosinecbcbdoheupsmaxups1'])

      recomBprmff100I10Lr0003R01Model.append(methodsI['RecomBprmff100I10Lr0003R01'])
      recomBprmff20I20Lr0003R01Model.append(methodsI['RecomBprmff20I20Lr0003R01'])

   recomTheMostPopularLabel:str = "Most pop."
   recomKnnLabel:str = "iKNN"
   recomVmcontextknn25Label:str = "SKNN k:25"
   recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Label:str = "Word2vec f:32, w:1, i:50K, agg:wAVG, len:3"
   recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Label:str = "Word2vec f:64, w:1, i:50K, agg:wAVG, len:7"
   recomCosinecbcbdoheupsweightedmeanups3Label:str = "Cosine CB agg:wAVG, len:3"
   recomCosinecbcbdoheupsmaxups1Label:str = "Cosine CB agg:max, len:1"
   recomBprmff100I10Lr0003R01Label: str = "BPR MF f:100, i:10, lr:0.003, reg:0.1"
   recomBprmff20I20Lr0003R01Label: str = "BPR MF f:20, i:20, lr:0.003, reg:0.1"

   windowSize:int = 801
   polynomialOrder:int = 1
   recomTheMostPopularModelSF:List[float] = savgol_filter(recomTheMostPopularModel, windowSize, polynomialOrder)  # window size 51, polynomial order 3
   recomKnnModelSF:List[float] = savgol_filter(recomKnnModel, windowSize, polynomialOrder)
   recomVmcontextknn25ModelSF:List[float] = savgol_filter(recomVmcontextknn25Model, windowSize, polynomialOrder)
   recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3SF:List[float] = savgol_filter(recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Model, windowSize, polynomialOrder)
   recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7SF:List[float] = savgol_filter(recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Model, windowSize, polynomialOrder)
   recomCosinecbcbdoheupsweightedmeanups3ModelSF:List[float] = savgol_filter(recomCosinecbcbdoheupsweightedmeanups3Model, windowSize, polynomialOrder)
   recomCosinecbcbdoheupsmaxups1SF:List[float] = savgol_filter(recomCosinecbcbdoheupsmaxups1Model, windowSize, polynomialOrder)
   recomBprmff100I10Lr0003R01ModelSF: List[float] = savgol_filter(recomBprmff100I10Lr0003R01Model, windowSize, polynomialOrder)
   recomBprmff20I20Lr0003R01ModelSF: List[float] = savgol_filter(recomBprmff20I20Lr0003R01Model, windowSize, polynomialOrder)


   y:List[int] = range(107538)
   #plt.figure(figsize=(4, 7))
   #plt.figure(figsize=(7, 3.5))
   plt.figure(figsize=(11, 6))


   plt.plot(recomTheMostPopularModel, label=recomTheMostPopularLabel, linestyle='-', linewidth=0.7)
   plt.plot(recomKnnModel, label=recomKnnLabel, linestyle='--', linewidth=0.7)
   plt.plot(recomVmcontextknn25Model, label=recomVmcontextknn25Label, linestyle='--', linewidth=0.7)
   plt.plot(recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Model, label=recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Label, linestyle=':', linewidth=0.7)
   plt.plot(recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Model, label=recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Label, linestyle=':', linewidth=0.7)
   plt.plot(recomCosinecbcbdoheupsweightedmeanups3Model, label=recomCosinecbcbdoheupsweightedmeanups3Label, linestyle='-.', linewidth=0.7)
   plt.plot(recomCosinecbcbdoheupsmaxups1Model, label=recomCosinecbcbdoheupsmaxups1Label, linestyle='-.', linewidth=0.7)
   plt.plot(recomBprmff100I10Lr0003R01Model, label=recomBprmff100I10Lr0003R01Label, linestyle='dashdot', linewidth=0.7)
   plt.plot(recomBprmff20I20Lr0003R01Model, label=recomBprmff20I20Lr0003R01Label, linestyle='dashdot', linewidth=0.7)
#
#   plt.gca().set_prop_cycle(None)

#   plt.plot(recomTheMostPopularModelSF, label=recomTheMostPopularLabel, linewidth=1.0)
#   plt.plot(recomKnnModelSF, label=recomKnnLabel, linewidth=1.0)
#   plt.plot(recomVmcontextknn25ModelSF, label=recomVmcontextknn25Label, linewidth=1.0)
#   plt.plot(recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3SF, label=recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Label, linewidth=1.0)
#   plt.plot(recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7SF, label=recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Label, linewidth=1.0)
#   plt.plot(recomCosinecbcbdoheupsweightedmeanups3ModelSF, label=recomCosinecbcbdoheupsweightedmeanups3Label, linewidth=1.0)
#   plt.plot(recomCosinecbcbdoheupsmaxups1SF, label=recomCosinecbcbdoheupsmaxups1Label, linewidth=1.0)
#   plt.plot(recomBprmff100I10Lr0003R01ModelSF, label=recomBprmff100I10Lr0003R01Label, linewidth=1.0)
#   plt.plot(recomBprmff20I20Lr0003R01ModelSF, label=recomBprmff20I20Lr0003R01Label, linewidth=1.0)

   #plt.xticks(rotation=45)
   #plt.yticks(rotation=90)

   #plt.ylabel('y - time', labelpad=-725)
   plt.xlabel('Votes assignment values')
   #plt.title(batchID + " - " + jobID)

   #plt.legend(bbox_to_anchor=(0.5, 0.99), loc=2, borderaxespad=0.)
   plt.legend()

   outputFileName:str = ".." + os.sep + "images" + os.sep + batchID + fileName.replace(".txt", ".png")

   #plt.figure(figsize=(10, 10))
   #plt.rcParams["figure.figsize"] = (20, 20)
   plt.savefig(outputFileName)
   #print("para" + batchID + fileName)
   plt.show()

   return

   fig, axs = plt.subplots(5)
   fig.suptitle(batchID + " - " + jobID)
   axs[0].plot(recomTheMostPopularModel, label=recomTheMostPopularLabel)
   axs[0].legend()
   axs[1].plot(recomKnnModel, label=recomKnnLabel)
   axs[1].legend()
   axs[2].plot(recomKnnModel, label=recomKnnLabel)
   axs[2].legend()

   axs[3].plot(recomVmcontextknn25Model, label=recomVmcontextknn25Label)
   axs[3].legend()
   axs[4].plot(recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Model, label=recomW2Vtpositivei50000Ws1Vs32Upsweightedmeanups3Label)
   axs[4].legend()
   axs[5].plot(recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Model, label=recomW2Vtpositivei50000Ws1Vs64Upsweightedmeanups7Label)
   axs[5].legend()
   axs[6].plot(recomCosinecbcbdoheupsweightedmeanups3Model, label=recomCosinecbcbdoheupsweightedmeanups3Label)
   axs[6].legend()
   axs[7].plot(recomCosinecbcbdoheupsmaxups1Model, label=recomCosinecbcbdoheupsmaxups1Label)
   axs[7].legend()

   axs[8].plot(recomBprmff100I10Lr0003R01Model, label=recomBprmff100I10Lr0003R01Label)
   axs[8].legend()
   axs[9].plot(recomBprmff20I20Lr0003R01Model, label=recomBprmff20I20Lr0003R01Label)
   axs[9].legend()



   #plt.show()
   #plt.savefig("para" + batchID + fileName)

   print(recomKnnModel[0:10])
   #print(recomW2vPosnegMaxModel[0:10])



def visualizationDHondtModelST():
   print("Visualization D'Hont Model")
   #os.chdir("..")

   #batchID:str = "stDiv90Ulinear0109R1"
   batchID:str = "stDiv90Ulinear0109R2"

   #fileName:str = "portfModelTimeEvolution-DHontFixed.txt"
   #fileName:str = "portfModelTimeEvolution-DHontRoulette.txt"
   #fileName:str = "portfModelTimeEvolution-DHontRoulette3.txt"

   fileName:str = "portfModelTimeEvolution-ContextDHondtRoulette1.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtRoulette3.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeFixed.txt"

   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin0500HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin07500HLin05025.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin1005HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin07500HLin07500.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin07500HLin075025.txt"
   #ileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin1005HLin1005.txt"

   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceOLin0500HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceOLin07500HLin05025.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceOLin1005HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceProbOLin07500HLin07500.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceProbOLin07500HLin075025.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceProbOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceProbOLin1005HLin1005.txt"

   jobID:str = fileName[fileName.index("-")+1:fileName.index(".")]
   print(jobID)

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
         print(methodLineI)
         if "0." in methodLineI:
            votesI:float = float(methodLineI[methodLineI.index("0."):-1])
         elif "-" in methodLineI:
            votesI:float = float(methodLineI[methodLineI.index("-"):-1])
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
      print(methodsI)
      recomTheMostPopularModel.append(methodsI['RecomThemostpopular'])
      recomKnnModel.append(methodsI['RecomKnn'])
      recomVmcontextknnModel.append(methodsI['RecomVmcontextknn'])
      recomBpmmff50I20Lr01R003Model.append(methodsI['RecomBpmmff50I20Lr01R003'])
      recomBpmmff20I50Lr01R001Model.append(methodsI['RecomBpmmff20I50Lr01R001'])
      recomCoscbonemean1Model.append(methodsI['RecomCoscbonemean1'])
      recomCoscboneweightedmean5Model.append(methodsI['RecomCoscboneweightedmean5'])
      recomW2Vtalli100000Ws1Vs32Upsmaxups1Model.append(methodsI['RecomW2Vtalli100000Ws1Vs32Upsmaxups1'])
      recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Model.append(methodsI['RecomW2Vtalli200000Ws1Vs64Upsweightedmeanups5'])

   recomTheMostPopularLabel:str = "Most pop."
   recomKnnLabel:str = "iKNN"
   recomVmcontextknnLabel:str = "SKNN k:25"
   recomBpmmff50I20Lr01R003Label:str = "BPR MF f:50, i:20, lr:0.1, reg:0.03"
   recomBpmmff20I50Lr01R001Label:str = "BPR MF f:20, i:50, lr:0.1, reg:0.01"
   recomCBmean1Label:str = "Cosine CB one agg:max, len:1"
   recomCBweightedmean5Label:str = "Cosine CB one agg:wAVG, len:5"
   recomW2Vtalli100000Ws1Vs32Upsmaxups1Label:str = "Word2vec f:32, w:1, i:100K, agg:max, len:1"
   recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Label:str = "Word2vec f:64, w:1, i:200K, agg:wAVG, len:5"

   windowSize:int = 801
   polynomialOrder:int = 1
   recomTheMostPopularModelSF:List[float] = savgol_filter(recomTheMostPopularModel, windowSize, polynomialOrder)  # window size 51, polynomial order 3
   recomKnnModelSF:List[float] = savgol_filter(recomKnnModel, windowSize, polynomialOrder)
   recomVmcontextknnModelSF:List[float] = savgol_filter(recomVmcontextknnModel, windowSize, polynomialOrder)
   recomBpmmff50I20Lr01R003ModelSF: List[float] = savgol_filter(recomBpmmff50I20Lr01R003Model, windowSize, polynomialOrder)
   recomBpmmff20I50Lr01R001ModelModelSF: List[float] = savgol_filter(recomBpmmff20I50Lr01R001Model, windowSize, polynomialOrder)
   recomCoscbonemean1ModelSF:List[float] = savgol_filter(recomCoscbonemean1Model, windowSize, polynomialOrder)
   recomCoscboneweightedmean5ModelSF:List[float] = savgol_filter(recomCoscboneweightedmean5Model, windowSize, polynomialOrder)
   recomW2Vtalli100000Ws1Vs32Upsmaxups1SF:List[float] = savgol_filter(recomW2Vtalli100000Ws1Vs32Upsmaxups1Model, windowSize, polynomialOrder)
   recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5SF:List[float] = savgol_filter(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Model, windowSize, polynomialOrder)

   y:List[int] = range(107538)
   #plt.figure(figsize=(4, 7))
   plt.figure(figsize=(11, 7))

   plt.plot(recomTheMostPopularModel, label=recomTheMostPopularLabel, linestyle='-', linewidth=0.7)
   plt.plot(recomKnnModel, label=recomKnnLabel, linestyle='--', linewidth=0.7)
   plt.plot(recomVmcontextknnModel, label=recomVmcontextknnLabel, linestyle='--', linewidth=0.7)
   plt.plot(recomBpmmff50I20Lr01R003Model, label=recomBpmmff50I20Lr01R003Label, linestyle=':', linewidth=0.7)
   plt.plot(recomBpmmff20I50Lr01R001Model, label=recomBpmmff20I50Lr01R001Label, linestyle=':', linewidth=0.7)
   plt.plot(recomCoscbonemean1Model, label=recomCBmean1Label, linestyle='-.', linewidth=0.7)
   plt.plot(recomCoscboneweightedmean5Model, label=recomCBweightedmean5Label, linestyle='-.', linewidth=0.7)
   plt.plot(recomW2Vtalli100000Ws1Vs32Upsmaxups1Model, label=recomW2Vtalli100000Ws1Vs32Upsmaxups1Label, linestyle='dashdot', linewidth=0.7)
   plt.plot(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Model, label=recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Label, linestyle='dashdot', linewidth=0.7)

   #plt.gca().set_prop_cycle(None)

   #plt.plot(recomTheMostPopularModelSF, label=recomTheMostPopularLabel, linewidth=1.0)
   #plt.plot(recomKnnModelSF, label=recomKnnLabel, linewidth=1.0)
   #plt.plot(recomVmcontextknnModelSF, label=recomVmcontextknnLabel, linewidth=0.02)
   #plt.plot(recomBpmmff50I20Lr01R003ModelSF, label=recomBpmmff50I20Lr01R003Label, linewidth=1.0)
   #plt.plot(recomBpmmff50I20Lr01R001ModelSF, label=recomBpmmff20I50Lr01R001Label, linewidth=1.0)
   #plt.plot(recomCoscbonemean1ModelSF, label=recomCBmean1Label, linewidth=1.0)
   #plt.plot(recomCoscboneweightedmean5ModelSF, label=recomCBweightedmean5Label, linewidth=1.0)
   #plt.plot(recomW2Vtalli100000Ws1Vs32Upsmaxups1SF, label=recomW2Vtalli100000Ws1Vs32Upsmaxups1Label, linewidth=1.0)
   #plt.plot(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5SF, label=recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5Label, linewidth=1.0)

   #plt.xticks(rotation=45)
   #plt.yticks(rotation=90)

   #plt.ylabel('y - time', labelpad=-725)
   plt.xlabel('Votes assignment values')
   #plt.title(batchID + " - " + jobID)

   plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.)

   outputFileName:str = ".." + os.sep + "images" + os.sep + batchID + fileName.replace(".txt", ".png")

   plt.savefig(outputFileName)
   #print("para" + batchID + fileName)
   #plt.savefig("/home/stepan/workspaceJup/HeterRecomPortfolio/images/" + "para" + batchID + fileName)
   plt.show()



   fig, axs = plt.subplots(5)
   fig.suptitle(batchID + " - " + jobID)
   axs[0].plot(recomTheMostPopularModel, label=recomTheMostPopularLabel)
   axs[0].legend()
   axs[1].plot(recomKnnModel, label=recomKnnLabel)
   axs[1].legend()
   axs[2].plot(recomBpmmff50I20Lr01R003Model, label=recomBpmmff50I20Lr01R003Label)
   axs[2].legend()
   axs[3].plot(recomBpmmff20I50Lr01R001Model, label=recomBpmmff20I50Lr01R001Label)
   axs[3].legend()
   axs[4].plot(recomCoscbonemean1Model, label=recomCBmean1Label)
   axs[4].legend()
   axs[5].plot(recomCoscboneweightedmean5Model, label=recomCBweightedmean5Label)
   axs[5].legend()


   #plt.show()
   #plt.savefig("para" + batchID + fileName)

   print(recomKnnModel[0:10])
   #print(recomW2vPosnegMaxModel[0:10])



def visualizationDHondtModelViolinPlotST():
   print("Visualization D'Hont Model")

   import seaborn.categorical
   seaborn.categorical._Old_Violin = seaborn.categorical._ViolinPlotter

   class _My_ViolinPlotter(seaborn.categorical._Old_Violin):

      def __init__(self, *args, **kwargs):
         super(_My_ViolinPlotter, self).__init__(*args, **kwargs)
         self.gray = 'black'
         self.linewidth = self.linewidth /2

   seaborn.categorical._ViolinPlotter = _My_ViolinPlotter

   import seaborn as sns
   import matplotlib.pyplot as plt


   #batchID:str = "stDiv90Ulinear0109R1"
   #batchID:str = "stDiv90Ulinear0109R2"
   #batchID:str = "stDiv90Upowerlaw054min048R2"
   batchID:str = "stDiv90Ustatic08R2"
   #batchID: str = "online"

   #fileName:str = "portfModelTimeEvolution-DHontFixed.txt"
   #fileName:str = "portfModelTimeEvolution-DHontRoulette.txt"
   #fileName:str = "portfModelTimeEvolution-DHontRoulette3.txt"

   fileName:str = "portfModelTimeEvolution-3.txt"
   fileName:str = "portfModelTimeEvolution-5.txt"

   fileName:str = "portfModelTimeEvolution-ContextDHondtRoulette1.txt"
   #fileName:str = "portfModelTimeEvolution-ContextDHondtRoulette3.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeFixed.txt"

   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin0500HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin07500HLin05025.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceOLin1005HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin07500HLin07500.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin07500HLin075025.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-ContextFDHondtDirectOptimizeINFFixedReduceProbOLin1005HLin1005.txt"

   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceOLin0500HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceOLin07500HLin05025.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceOLin1005HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceProbOLin07500HLin07500.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceProbOLin07500HLin075025.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceProbOLin1000HLin1005.txt"
   #fileName:str = "portfModelTimeEvolution-FuzzyDHondtINFFixedClk003ViewDivisor250ReduceProbOLin1005HLin1005.txt"

   jobID:str = fileName[fileName.index("-")+1:fileName.index(".")]
   print(jobID)

   inputFileName:str = Configuration.resultsDirectory + os.sep + batchID + os.sep + fileName
   #inputFileName:str = "../resultsOnline" + os.sep + fileName

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
         #print(methodLineI)
         if "0." in methodLineI:
            votesI:float = float(methodLineI[methodLineI.index("0."):-1])
         elif "-" in methodLineI:
            votesI:float = float(methodLineI[methodLineI.index("-"):-1])
         methodsI[methodIDI] = votesI
         methodLineI = f.readline()

      currentItemIDs.append(currentItemID)
      userIDs.append(userID)
      methods.append(methodsI)

   methodIDsList:List = []
   weightsList:List = []

   recomThemostpopularList:List = []
   recomKnnList:List = []
   recomVmcontextknnList:List = []
   recomBpmmff50I20Lr01R003List:List = []
   recomBpmmff20I50Lr01R001List:List = []
   recomCoscbonemean1List:List = []
   recomCoscboneweightedmean5List:List = []
   recomW2Vtalli100000Ws1Vs32Upsmaxups1List:List = []
   recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List:List = []

   for methodI in methods:
      for methodIdI, weightI in methodI.items():
         if weightI < 0:
            weightI = 0

         if methodIdI == 'RecomThemostpopular':
            recomThemostpopularList.append(weightI)
         elif methodIdI == 'RecomKnn':
            recomKnnList.append(weightI)
         elif methodIdI == 'RecomVmcontextknn':
            recomVmcontextknnList.append(weightI)
         elif methodIdI == 'RecomBpmmff50I20Lr01R003':
            recomBpmmff50I20Lr01R003List.append(weightI)
         elif methodIdI == 'RecomBpmmff20I50Lr01R001':
            recomBpmmff20I50Lr01R001List.append(weightI)
         elif methodIdI == 'RecomCoscbonemean1':
            recomCoscbonemean1List.append(weightI)
         elif methodIdI == 'RecomCoscboneweightedmean5':
            recomCoscboneweightedmean5List.append(weightI)
         elif methodIdI == 'RecomW2Vtalli100000Ws1Vs32Upsmaxups1':
            recomW2Vtalli100000Ws1Vs32Upsmaxups1List.append(weightI)
         elif methodIdI == 'RecomW2Vtalli200000Ws1Vs64Upsweightedmeanups5':
            recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List.append(weightI)

   # remove the first 10%
   recomThemostpopularList = recomThemostpopularList[int(0.1*len(recomThemostpopularList)):]
   recomKnnList = recomKnnList[int(0.1*len(recomKnnList)):]
   recomVmcontextknnList = recomVmcontextknnList[int(0.1*len(recomVmcontextknnList)):]
   recomBpmmff50I20Lr01R003List = recomBpmmff50I20Lr01R003List[int(0.1*len(recomBpmmff50I20Lr01R003List)):]
   recomBpmmff20I50Lr01R001List = recomBpmmff20I50Lr01R001List[int(0.1*len(recomBpmmff20I50Lr01R001List)):]
   recomCoscbonemean1List = recomCoscbonemean1List[int(0.1*len(recomCoscbonemean1List)):]
   recomCoscboneweightedmean5List = recomCoscboneweightedmean5List[int(0.1*len(recomCoscboneweightedmean5List)):]
   recomW2Vtalli100000Ws1Vs32Upsmaxups1List = recomW2Vtalli100000Ws1Vs32Upsmaxups1List[int(0.1*len(recomW2Vtalli100000Ws1Vs32Upsmaxups1List)):]
   recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List = recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List[int(0.1*len(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List)):]


   recomThemostpopularList.sort()
   recomThemostpopularList = recomThemostpopularList[int(0.05*len(recomThemostpopularList)):int(0.95*len(recomThemostpopularList))]
   #print(recomThemostpopularList)

   recomKnnList.sort()
   recomKnnList = recomKnnList[int(0.05*len(recomKnnList)):int(0.95*len(recomKnnList))]

   recomVmcontextknnList.sort()
   recomVmcontextknnList = recomVmcontextknnList[int(0.05*len(recomVmcontextknnList)):int(0.95*len(recomVmcontextknnList))]

   recomBpmmff50I20Lr01R003List.sort()
   recomBpmmff50I20Lr01R003List = recomBpmmff50I20Lr01R003List[int(0.05*len(recomBpmmff50I20Lr01R003List)):int(0.95*len(recomBpmmff50I20Lr01R003List))]

   recomBpmmff20I50Lr01R001List.sort()
   recomBpmmff20I50Lr01R001List = recomBpmmff20I50Lr01R001List[int(0.05*len(recomBpmmff20I50Lr01R001List)):int(0.95*len(recomBpmmff20I50Lr01R001List))]

   recomCoscbonemean1List.sort()
   recomCoscbonemean1List = recomCoscbonemean1List[int(0.05*len(recomCoscbonemean1List)):int(0.95*len(recomCoscbonemean1List))]

   recomCoscboneweightedmean5List.sort()
   recomCoscboneweightedmean5List = recomCoscboneweightedmean5List[int(0.05*len(recomCoscboneweightedmean5List)):int(0.95*len(recomCoscboneweightedmean5List))]

   recomW2Vtalli100000Ws1Vs32Upsmaxups1List.sort()
   recomW2Vtalli100000Ws1Vs32Upsmaxups1List = recomW2Vtalli100000Ws1Vs32Upsmaxups1List[int(0.05*len(recomW2Vtalli100000Ws1Vs32Upsmaxups1List)):int(0.95*len(recomW2Vtalli100000Ws1Vs32Upsmaxups1List))]

   recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List.sort()
   recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List = recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List[int(0.05*len(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List)):int(0.95*len(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List))]


   methodIDsList.extend(["Most pop."]*len(recomThemostpopularList))
   weightsList.extend(recomThemostpopularList)

   methodIDsList.extend(["iKNN"]*len(recomKnnList))
   weightsList.extend(recomKnnList)

   methodIDsList.extend(["SKNN k:25"] * len(recomVmcontextknnList))
   weightsList.extend(recomVmcontextknnList)

   methodIDsList.extend(["BPR MF f:50, i:20, lr:0.1, reg:0.03"] * len(recomBpmmff50I20Lr01R003List))
   weightsList.extend(recomBpmmff50I20Lr01R003List)

   methodIDsList.extend(["BPR MF f:20, i:50, lr:0.1, reg:0.01"] * len(recomBpmmff20I50Lr01R001List))
   weightsList.extend(recomBpmmff20I50Lr01R001List)

   methodIDsList.extend(["Cosine CB agg:max, len:1"] * len(recomCoscbonemean1List))
   weightsList.extend(recomCoscbonemean1List)

   methodIDsList.extend(["Cosine CB agg:wAVG, len:5"] * len(recomCoscboneweightedmean5List))
   weightsList.extend(recomCoscboneweightedmean5List)

   methodIDsList.extend(["Word2vec f:32, w:1, i:100K, agg:max, len:1"] * len(recomW2Vtalli100000Ws1Vs32Upsmaxups1List))
   weightsList.extend(recomW2Vtalli100000Ws1Vs32Upsmaxups1List)

   methodIDsList.extend(["Word2vec f:64, w:1, i:200K, agg:wAVG, len:5"] * len(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List))
   weightsList.extend(recomW2Vtalli200000Ws1Vs64Upsweightedmeanups5List)

   methodIDsList.reverse()
   weightsList.reverse()

   a = DataFrame(
      {'': methodIDsList,
       'Votes assignment': weightsList
       })

   # Create DataFrame
   df:DataFrame = DataFrame(methods)
   #print(df)

   plt.figure(figsize=(12, 7))

   colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22']
   colors.reverse()

   #ax = sns.violinplot(x=tips["total_bill"])
   #ax = sns.violinplot(x="day", y="total_bill", data=tips)
   ax = sns.violinplot(x='', y="Votes assignment", data=a, palette=colors)
   #ax.set_xlabel('XLabel', loc='right')



   outputFileName:str = ".." + os.sep + "images" + os.sep + "obs" + batchID + fileName.replace(".txt", ".png")


   plt.xticks(rotation=-12, ha="left")

   plt.gcf().subplots_adjust(bottom=0.35, left=0.15)
   #plt.show()
   plt.savefig(outputFileName)


if __name__ == "__main__":
   os.chdir("..")

   #visualizationDHondtModelML()
   #visualizationDHondtModelST()

   visualizationDHondtModelViolinPlotST()