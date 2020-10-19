#!/usr/bin/python3

import time
import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

from configuration.configuration import Configuration #class

import random
import numpy as np

from typing import List

class Names:

    portIDs:List[str] = ["DHontFixed", "DHontRoulette1", "DHontRoulette3"]
    portNegIDs:List[str] = ["NegDHontFixed", "NegDHontRoulette1", "NegDHontRoulette3"]

    ortNegDHontIDs:List[str] = ["NegDHontFixedClk01ViewDivisor200", "NegDHontRoulette1Clk01ViewDivisor200", "NegDHontRoulette3Clk01ViewDivisor200"]
    portNegDhontThSampIDs:List[str] = ["NegDHondtThompsonSamplingFixed", "NegDHondtThompsonSamplingRoulette1", "NegDHondtThompsonSamplingRoulette3"]


    negFeedback:List[str] = ["OLin0802HLin1002", "OStat08HLin1002"]

    lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
    # lrClicks: List[float] = [0.1]
    #lrViewDivisors:List[float] = [0.0002, 0.0005, 0.0001]
    lrViewDivisors:List[float] = [200, 500, 1000]
    # lrViewDivisors: List[float] = [500]


def visualizationBatches():

    visualizationMatrixClkViewDivisor("ml1mDiv90Ustatic08R1", Names.portIDs)
    visualizationMatrixClkViewDivisor("ml1mDiv90Ustatic06R1", Names.portIDs)
    visualizationMatrixClkViewDivisor("ml1mDiv90Ustatic04R1", Names.portIDs)
    visualizationMatrixClkViewDivisor("ml1mDiv90Ustatic02R1", Names.portIDs)
    visualizationMatrixClkViewDivisor("ml1mDiv90Ulinear0109R1", Names.portIDs)

    visualizationMatrixClkViewDivisor("ml1mDiv90Ustatic08R2", Names.portIDs)
    visualizationMatrixClkViewDivisor("ml1mDiv90Ustatic06R2", Names.portIDs)
    visualizationMatrixClkViewDivisor("ml1mDiv90Ustatic04R2", Names.portIDs)
    visualizationMatrixClkViewDivisor("ml1mDiv90Ustatic02R2", Names.portIDs)
    visualizationMatrixClkViewDivisor("ml1mDiv90Ulinear0109R2", Names.portIDs)

    visualizationPortfoliosAndNegImplFeedback("ml1mDiv90Ulinear0109R5", Names.negFeedback, Names.portNegDhontThSampIDs + Names.ortNegDHontIDs)
    visualizationPortfoliosAndNegImplFeedback("ml1mDiv90Ustatic08R5", Names.negFeedback, Names.portNegDhontThSampIDs + Names.ortNegDHontIDs)


def visualizationMatrixClkViewDivisor(batchID:str, portIDs:List[str]):

    print("VisualizationMatrix")


    lrClicksLegend: List[str] = ["lrClicks " + str(lrClicksI) for lrClicksI in Names.lrClicks]
    lrViewLegend: List[str] = ["lrView " + str(lrViewI) for lrViewI in Names.lrViewDivisors]

    dataStrI:str
    batchIDI:str
    portIDSuffixI:str

    for portIDI in portIDs:

        dataStrI:str = readFile(Configuration.resultsDirectory + os.sep + batchID + os.sep + "evaluation.txt")

        resultsIJ:List[List[int]] = readData_(dataStrI, portIDI, "", Names.lrClicks, Names.lrViewDivisors)

        generateMatrix(batchID, portIDI, resultsIJ, lrViewLegend, lrClicksLegend)


def visualizationPortfoliosAndNegImplFeedback(batchIDI:str, negImplFeedback:List[str], portNames:List[str]):

    dataStr:str = readFile(Configuration.resultsDirectory + os.sep + batchIDI + os.sep + "evaluation.txt")

    matrixOfKeys:List[List[str]] = []
    for portNegDhontThSampIDsJ in portNames:
        rowI:List[str] = []
        for negFeedbackI in negImplFeedback:
            textIndexIJ:str = portNegDhontThSampIDsJ + negFeedbackI
            rowI.append(textIndexIJ)
        matrixOfKeys.append(rowI)

    print(matrixOfKeys)

    data = parseData(dataStr, matrixOfKeys)
    print(data)


    generateMatrix(batchIDI, "", data, Names.negFeedback, portNames)


def generateMatrix(batchID:str, portID:str, results:List[List[int]], xLegend:List[float], yLegend:List[float]):


    harvest = np.array(results)

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xLegend)))
    ax.set_yticks(np.arange(len(yLegend)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xLegend)
    ax.set_yticklabels(yLegend)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(yLegend)):
      for j in range(len(xLegend)):
          text = ax.text(j, i, harvest[i, j],
                         ha="center", va="center", color="w")

    ax.set_title("Results " + batchID + "-" + portID)
    fig.tight_layout()
    #plt.show()

    plt.savefig(Configuration.imagesDirectory + os.sep + batchID + "-" + portID + '.png')


def readData_(dataStr:str, portID:str, portIDSuffix:str, lrClicks:List[float], lrViewDivisors:List[float]):

    resultsDict:dict = {}

    for resultLineI in dataStr.split("ids: ['"):
        portfolioNameI:str = resultLineI[0:resultLineI.find("']")]
        clicksI:str = resultLineI[resultLineI.find("[[{'clicks': ") + len("[[{'clicks': "):resultLineI.find("}]]")]

        if clicksI == '':
            continue

        resultsDict[portfolioNameI] = int(clicksI)
    print(resultsDict)

    csv:str = ""
    results:List[List[int]] = []

    for lrClickI in lrClicks:
        rowI = []
        for lrViewDivisorJ in lrViewDivisors:
            portfolioIDI:str = portID + "Clk" + str(lrClickI).replace(".", "") + "ViewDivisor" + str(lrViewDivisorJ).replace(".", "") + portIDSuffix
            #clicksI:int = resultsDict.get(portfolioIDI, None)
            print(portfolioIDI)
            clicksI:int = resultsDict[portfolioIDI]

            csv += str(clicksI) + ";"
            rowI.append(clicksI)
        csv += '\n'
        results.append(rowI)
    #print(csv)
    #print(results)

    return results



def parseData(dataStr:str, matrixOfKeys:List[List[str]]):

    resultsDict:dict = {}

    for resultLineI in dataStr.split("ids: ['"):
        portfolioNameI:str = resultLineI[0:resultLineI.find("']")]
        clicksI:str = resultLineI[resultLineI.find("[[{'clicks': ") + len("[[{'clicks': "):resultLineI.find("}]]")]

        if clicksI == '':
            continue

        resultsDict[portfolioNameI] = int(clicksI)

    data:List[List[int]] = []
    rowI:List[str]
    for rowI in matrixOfKeys:
        dataRowI:List[int] = []
        for keyJ in rowI:
            valueIJ:int = resultsDict[keyJ]
            dataRowI.append(valueIJ)
        data.append(dataRowI)

    return data



def readFile(fileName:str):
    data:str = None
    #with open (fileName, "r") as myfile:
    #    data=myfile.readlines()
    data = open(fileName, 'r').read()
    return data



if __name__ == "__main__":

  np.random.seed(42)
  random.seed(42)

  os.chdir("..")

  visualizationBatches()