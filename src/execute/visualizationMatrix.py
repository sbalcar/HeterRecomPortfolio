#!/usr/bin/python3

import time
import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

import random
import numpy as np

from typing import List


def visualizationMatrix():

    np.random.seed(42)
    random.seed(42)

    print("VisualizationMatrix")

    portIDs:List[str] = ["DHontFixed", "DHontRoulette1", "DHontRoulette3"]
    portNegIDs:List[str] = ["NegDHontFixed", "NegDHontRoulette1", "NegDHontRoulette3"]


    lrClicks:List[float] = [0.2, 0.1, 0.02, 0.005]
    # lrClicks: List[float] = [0.1]
    #lrViewDivisors:List[float] = [0.0002, 0.0005, 0.0001]
    lrViewDivisors:List[float] = [200, 500, 1000]
    # lrViewDivisors: List[float] = [500]

    a:str = [
             #("ml1mDiv90Ustatic02R1", ml1mDiv90Ustatic02R1, portIDs, ""),
             #("ml1mDiv90Ustatic02R1", ml1mDiv90Ustatic02R1, portNegIDs, "OLin0802HLin1002"),
             #("ml1mDiv90Ustatic02R1", ml1mDiv90Ustatic02R1, portNegIDs, "OStat08HLin1002"),

             #("ml1mDiv90Ustatic08R1", readFile("ml1mDiv90Ustatic08R1.txt"), portIDs, ""),
             #("ml1mDiv90Ustatic06R1", readFile("ml1mDiv90Ustatic06R1.txt"), portIDs, ""),
             #("ml1mDiv90Ustatic04R1", readFile("ml1mDiv90Ustatic04R1.txt"), portIDs, ""),
             #("ml1mDiv90Ustatic02R1", readFile("ml1mDiv90Ustatic02R1.txt"), portIDs, ""),
             #("ml1mDiv90Ulinear0109R1", readFile("ml1mDiv90Ulinear0109R1.txt"), portIDs, ""),



             #("ml1mDiv90Ustatic08R1", ml1mDiv90Ustatic08R1, portIDs, ""),
             #("ml1mDiv90Ustatic08R1", ml1mDiv90Ustatic08R1, portNegIDs, "OLin0802HLin1002"),
             #("ml1mDiv90Ustatic08R1", ml1mDiv90Ustatic08R1, portNegIDs, "OStat08HLin1002")

    ]

    dataStrI:str
    batchIDI:str
    portIDSuffixI:str

    for batchIDI, dataStrI, portIDs, portIDSuffixI in a:
        for portIDI in portIDs:

            resultsIJ:List[List[int]] = readData(dataStrI, portIDI, portIDSuffixI, lrClicks, lrViewDivisors)

            generateMatrix(batchIDI, portIDI, portIDSuffixI, resultsIJ, lrClicks, lrViewDivisors)



def readData(dataStr:str, portID:str, portIDSuffix:str, lrClicks:List[float], lrViewDivisors:List[float]):

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


def generateMatrix(batchID:str, portID:str, portIDSuffix:str, results:List[List[int]], lrClicks:List[float], lrViewDivisors:List[float]):

    lrClicksLegend:List[str] = ["lrClicks " + str(lrClicksI) for lrClicksI in lrClicks]
    lrViewLegend:List[str] = ["lrView " + str(lrViewI) for lrViewI in lrViewDivisors]

    harvest = np.array(results)

    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(lrViewLegend)))
    ax.set_yticks(np.arange(len(lrClicksLegend)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(lrViewLegend)
    ax.set_yticklabels(lrClicksLegend)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(lrClicksLegend)):
      for j in range(len(lrViewLegend)):
          text = ax.text(j, i, harvest[i, j],
                         ha="center", va="center", color="w")

    ax.set_title("Results " + batchID + "-" + portID + portIDSuffix)
    fig.tight_layout()
    #plt.show()

    plt.savefig(batchID + "-" + portID + portIDSuffix + '.png')


def readFile(fileName:str):
    data:str = None
    #with open (fileName, "r") as myfile:
    #    data=myfile.readlines()
    data = open(fileName, 'r').read()
    return data




visualizationMatrix()