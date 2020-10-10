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
    lrViewDivisors:List[float] = [0.0002, 0.0005, 0.0001]
    #lrViewDivisors:List[float] = [200, 500, 1000]
    # lrViewDivisors: List[float] = [500]

    a:str = [
             ("ml1mDiv90Ustatic02R1", ml1mDiv90Ustatic02R1, portIDs, ""),
             ("ml1mDiv90Ustatic02R1", ml1mDiv90Ustatic02R1, portNegIDs, "OLin0802HLin1002"),
             ("ml1mDiv90Ustatic02R1", ml1mDiv90Ustatic02R1, portNegIDs, "OStat08HLin1002")

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

    csv:str = ""
    results:List[List[int]] = []

    for lrClickI in lrClicks:
        rowI = []
        for lrViewDivisorJ in lrViewDivisors:
            portfolioIDI:str = portID + "Clk" + str(lrClickI).replace(".", "") + "View" + str(lrViewDivisorJ).replace(".", "") + portIDSuffix
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



ml1mDiv90Ustatic02R1:str = """ids: ['BanditTS']
[[{'clicks': 991}]]

ids: ['DHontFixedClk02View00002']
[[{'clicks': 1391}]]

ids: ['DHontFixedClk002View00002']
[[{'clicks': 1304}]]

ids: ['DHontFixedClk01View00002']
[[{'clicks': 1384}]]

ids: ['DHontFixedClk0005View00002']
[[{'clicks': 1317}]]

ids: ['DHontFixedClk01View00005']
[[{'clicks': 1323}]]

ids: ['DHontFixedClk02View00005']
[[{'clicks': 1386}]]

ids: ['DHontFixedClk0005View00005']
[[{'clicks': 1321}]]

ids: ['DHontFixedClk0005View00001']
[[{'clicks': 1316}]]

ids: ['DHontFixedClk002View00005']
[[{'clicks': 1312}]]

ids: ['DHontFixedClk002View00001']
[[{'clicks': 1300}]]

ids: ['DHontFixedClk01View00001']
[[{'clicks': 1310}]]

ids: ['DHontFixedClk02View00001']
[[{'clicks': 1276}]]

ids: ['DHontRoulette1Clk01View00002']
[[{'clicks': 786}]]

ids: ['DHontRoulette1Clk01View00001']
[[{'clicks': 1014}]]

ids: ['DHontRoulette1Clk0005View00001']
[[{'clicks': 770}]]

ids: ['DHontRoulette1Clk0005View00005']
[[{'clicks': 763}]]

ids: ['DHontRoulette1Clk01View00005']
[[{'clicks': 754}]]

ids: ['DHontRoulette1Clk02View00001']
[[{'clicks': 1191}]]

ids: ['DHontRoulette1Clk0005View00002']
[[{'clicks': 764}]]

ids: ['DHontRoulette1Clk002View00005']
[[{'clicks': 764}]]

ids: ['DHontRoulette1Clk002View00001']
[[{'clicks': 768}]]

ids: ['DHontRoulette3Clk02View00001']
[[{'clicks': 1140}]]

ids: ['DHontRoulette1Clk002View00002']
[[{'clicks': 766}]]

ids: ['DHontRoulette3Clk02View00002']
[[{'clicks': 1319}]]

ids: ['DHontRoulette3Clk01View00002']
[[{'clicks': 1120}]]

ids: ['DHontRoulette3Clk002View00002']
[[{'clicks': 1079}]]

ids: ['DHontRoulette1Clk02View00005']
[[{'clicks': 770}]]

ids: ['DHontRoulette1Clk02View00002']
[[{'clicks': 1009}]]

ids: ['DHontRoulette3Clk01View00001']
[[{'clicks': 1178}]]

ids: ['DHontRoulette3Clk01View00005']
[[{'clicks': 1056}]]

ids: ['DHontRoulette3Clk0005View00002']
[[{'clicks': 1059}]]

ids: ['DHontRoulette3Clk0005View00001']
[[{'clicks': 1073}]]

ids: ['DHontRoulette3Clk002View00005']
[[{'clicks': 1053}]]

ids: ['DHontRoulette3Clk02View00005']
[[{'clicks': 1160}]]

ids: ['DHontRoulette3Clk0005View00005']
[[{'clicks': 1069}]]

ids: ['DHontRoulette3Clk002View00001']
[[{'clicks': 1075}]]

ids: ['theMostPopular']
[[{'clicks': 1306}]]

ids: ['cosCBwindow3']
[[{'clicks': 1194}]]

ids: ['cosCBmax']
[[{'clicks': 409}]]

ids: ['w2vPosnegWindow3']
[[{'clicks': 533}]]

ids: ['NegDHontFixedClk02View00005OLin0802HLin1002']
[[{'clicks': 1228}]]

ids: ['NegDHontFixedClk02View00001OLin0802HLin1002']
[[{'clicks': 1280}]]

ids: ['NegDHontFixedClk02View00002OLin0802HLin1002']
[[{'clicks': 1365}]]

ids: ['NegDHontFixedClk01View00002OLin0802HLin1002']
[[{'clicks': 1229}]]

ids: ['NegDHontFixedClk002View00005OStat08HLin1002']
[[{'clicks': 1156}]]

ids: ['NegDHontFixedClk02View00005OStat08HLin1002']
[[{'clicks': 1170}]]

ids: ['NegDHontFixedClk02View00002OStat08HLin1002']
[[{'clicks': 1223}]]

ids: ['NegDHontFixedClk002View00002OLin0802HLin1002']
[[{'clicks': 1194}]]

ids: ['NegDHontFixedClk002View00001OLin0802HLin1002']
[[{'clicks': 1188}]]

ids: ['NegDHontFixedClk01View00001OStat08HLin1002']
[[{'clicks': 1330}]]

ids: ['NegDHontFixedClk01View00005OStat08HLin1002']
[[{'clicks': 1109}]]

ids: ['NegDHontFixedClk0005View00002OLin0802HLin1002']
[[{'clicks': 1203}]]

ids: ['NegDHontFixedClk0005View00001OLin0802HLin1002']
[[{'clicks': 1188}]]

ids: ['NegDHontFixedClk01View00005OLin0802HLin1002']
[[{'clicks': 1199}]]

ids: ['NegDHontFixedClk0005View00005OLin0802HLin1002']
[[{'clicks': 1169}]]

ids: ['NegDHontFixedClk01View00001OLin0802HLin1002']
[[{'clicks': 1323}]]

ids: ['NegDHontFixedClk02View00001OStat08HLin1002']
[[{'clicks': 797}]]

ids: ['NegDHontFixedClk002View00002OStat08HLin1002']
[[{'clicks': 1163}]]

ids: ['NegDHontFixedClk002View00005OLin0802HLin1002']
[[{'clicks': 1177}]]

ids: ['NegDHontFixedClk01View00002OStat08HLin1002']
[[{'clicks': 1164}]]

ids: ['NegDHontFixedClk0005View00002OStat08HLin1002']
[[{'clicks': 1150}]]

ids: ['NegDHontFixedClk0005View00001OStat08HLin1002']
[[{'clicks': 1139}]]

ids: ['NegDHontRoulette1Clk01View00002OLin0802HLin1002']
[[{'clicks': 768}]]

ids: ['NegDHontRoulette1Clk02View00005OLin0802HLin1002']
[[{'clicks': 803}]]

ids: ['NegDHontFixedClk0005View00005OStat08HLin1002']
[[{'clicks': 1152}]]

ids: ['NegDHontRoulette1Clk002View00001OLin0802HLin1002']
[[{'clicks': 743}]]

ids: ['NegDHontRoulette1Clk01View00005OLin0802HLin1002']
[[{'clicks': 762}]]

ids: ['NegDHontRoulette1Clk02View00002OLin0802HLin1002']
[[{'clicks': 821}]]

ids: ['NegDHontRoulette1Clk0005View00002OLin0802HLin1002']
[[{'clicks': 745}]]

ids: ['NegDHontRoulette1Clk01View00001OLin0802HLin1002']
[[{'clicks': 978}]]

ids: ['NegDHontRoulette1Clk002View00002OLin0802HLin1002']
[[{'clicks': 781}]]

ids: ['NegDHontRoulette1Clk002View00005OLin0802HLin1002']
[[{'clicks': 738}]]

ids: ['NegDHontRoulette1Clk02View00001OLin0802HLin1002']
[[{'clicks': 1156}]]

ids: ['NegDHontFixedClk002View00001OStat08HLin1002']
[[{'clicks': 1168}]]

ids: ['NegDHontRoulette1Clk02View00001OStat08HLin1002']
[[{'clicks': 1062}]]

ids: ['NegDHontRoulette1Clk0005View00001OLin0802HLin1002']
[[{'clicks': 778}]]

ids: ['NegDHontRoulette1Clk02View00005OStat08HLin1002']
[[{'clicks': 744}]]

ids: ['NegDHontRoulette1Clk01View00005OStat08HLin1002']
[[{'clicks': 734}]]

ids: ['NegDHontRoulette1Clk01View00002OStat08HLin1002']
[[{'clicks': 733}]]

ids: ['NegDHontRoulette1Clk002View00002OStat08HLin1002']
[[{'clicks': 740}]]

ids: ['NegDHontRoulette1Clk02View00002OStat08HLin1002']
[[{'clicks': 821}]]

ids: ['NegDHontRoulette1Clk0005View00002OStat08HLin1002']
[[{'clicks': 761}]]

ids: ['NegDHontRoulette1Clk0005View00001OStat08HLin1002']
[[{'clicks': 766}]]

ids: ['NegDHontRoulette1Clk0005View00005OLin0802HLin1002']
[[{'clicks': 792}]]

ids: ['NegDHontRoulette1Clk01View00001OStat08HLin1002']
[[{'clicks': 921}]]

ids: ['NegDHontRoulette3Clk02View00005OLin0802HLin1002']
[[{'clicks': 1032}]]

ids: ['NegDHontRoulette3Clk01View00002OLin0802HLin1002']
[[{'clicks': 1008}]]

ids: ['NegDHontRoulette1Clk0005View00005OStat08HLin1002']
[[{'clicks': 736}]]

ids: ['NegDHontRoulette3Clk02View00001OLin0802HLin1002']
[[{'clicks': 1192}]]

ids: ['NegDHontRoulette3Clk02View00002OLin0802HLin1002']
[[{'clicks': 1162}]]

ids: ['NegDHontRoulette1Clk002View00001OStat08HLin1002']
[[{'clicks': 737}]]

ids: ['NegDHontRoulette1Clk002View00005OStat08HLin1002']
[[{'clicks': 716}]]

ids: ['NegDHontRoulette3Clk02View00002OStat08HLin1002']
[[{'clicks': 1071}]]

ids: ['NegDHontRoulette3Clk01View00005OLin0802HLin1002']
[[{'clicks': 953}]]

ids: ['NegDHontRoulette3Clk01View00001OLin0802HLin1002']
[[{'clicks': 1182}]]

ids: ['NegDHontRoulette3Clk0005View00001OLin0802HLin1002']
[[{'clicks': 926}]]

ids: ['NegDHontRoulette3Clk002View00002OLin0802HLin1002']
[[{'clicks': 955}]]

ids: ['NegDHontRoulette3Clk002View00005OLin0802HLin1002']
[[{'clicks': 946}]]

ids: ['NegDHontRoulette3Clk002View00001OLin0802HLin1002']
[[{'clicks': 942}]]

ids: ['NegDHontRoulette3Clk02View00001OStat08HLin1002']
[[{'clicks': 1064}]]

ids: ['NegDHontRoulette3Clk0005View00005OLin0802HLin1002']
[[{'clicks': 983}]]

ids: ['NegDHontRoulette3Clk02View00005OStat08HLin1002']
[[{'clicks': 966}]]

ids: ['NegDHontRoulette3Clk002View00001OStat08HLin1002']
[[{'clicks': 868}]]

ids: ['NegDHontRoulette3Clk01View00002OStat08HLin1002']
[[{'clicks': 945}]]

ids: ['NegDHontRoulette3Clk002View00002OStat08HLin1002']
[[{'clicks': 887}]]

ids: ['NegDHontRoulette3Clk01View00005OStat08HLin1002']
[[{'clicks': 907}]]

ids: ['NegDHontRoulette3Clk0005View00002OLin0802HLin1002']
[[{'clicks': 957}]]

ids: ['NegDHontRoulette3Clk002View00005OStat08HLin1002']
[[{'clicks': 939}]]

ids: ['NegDHontRoulette3Clk01View00001OStat08HLin1002']
[[{'clicks': 1140}]]

ids: ['NegDHontRoulette3Clk0005View00005OStat08HLin1002']
[[{'clicks': 909}]]

ids: ['NegDHontRoulette3Clk0005View00002OStat08HLin1002']
[[{'clicks': 922}]]

ids: ['NegDHontRoulette3Clk0005View00001OStat08HLin1002']
[[{'clicks': 897}]]
"""



ml1mDiv90Ustatic08R1:str = """ids: ['theMostPopular']
[[{'clicks': 2578}]]

ids: ['cosCBwindow3']
[[{'clicks': 3627}]]

ids: ['cosCBmax']
[[{'clicks': 939}]]

ids: ['w2vPosnegWindow3']
[[{'clicks': 1900}]]

ids: ['BanditTS']
[[{'clicks': 3544}]]

ids: ['NegDHontRoulette1Clk01View00005OLin0802HLin1002']
[[{'clicks': 3103}]]

ids: ['NegDHontRoulette1Clk0005View00002OLin0802HLin1002']
[[{'clicks': 2996}]]

ids: ['NegDHontRoulette1Clk002View00002OLin0802HLin1002']
[[{'clicks': 2928}]]

ids: ['NegDHontRoulette1Clk002View00002OStat08HLin1002']
[[{'clicks': 3002}]]

ids: ['NegDHontRoulette1Clk01View00002OLin0802HLin1002']
[[{'clicks': 3930}]]

ids: ['NegDHontRoulette1Clk01View00005OStat08HLin1002']
[[{'clicks': 3151}]]

ids: ['NegDHontRoulette1Clk02View00002OStat08HLin1002']
[[{'clicks': 4359}]]

ids: ['NegDHontRoulette1Clk002View00001OLin0802HLin1002']
[[{'clicks': 3011}]]

ids: ['NegDHontRoulette1Clk01View00001OStat08HLin1002']
[[{'clicks': 4676}]]

ids: ['NegDHontRoulette1Clk02View00005OStat08HLin1002']
[[{'clicks': 3659}]]

ids: ['NegDHontRoulette1Clk002View00001OStat08HLin1002']
[[{'clicks': 2950}]]

ids: ['NegDHontRoulette1Clk002View00005OLin0802HLin1002']
[[{'clicks': 2961}]]

ids: ['NegDHontRoulette3Clk01View00002OLin0802HLin1002']
[[{'clicks': 4138}]]

ids: ['NegDHontRoulette1Clk02View00001OStat08HLin1002']
[[{'clicks': 4702}]]

ids: ['NegDHontRoulette1Clk01View00001OLin0802HLin1002']
[[{'clicks': 2384}]]

ids: ['NegDHontRoulette1Clk0005View00001OLin0802HLin1002']
[[{'clicks': 3013}]]

ids: ['NegDHontRoulette1Clk01View00002OStat08HLin1002']
[[{'clicks': 4488}]]

ids: ['NegDHontRoulette1Clk002View00005OStat08HLin1002']
[[{'clicks': 2972}]]

ids: ['NegDHontRoulette1Clk0005View00001OStat08HLin1002']
[[{'clicks': 2920}]]

ids: ['NegDHontRoulette3Clk02View00001OLin0802HLin1002']
[[{'clicks': 4080}]]

ids: ['NegDHontRoulette3Clk01View00001OLin0802HLin1002']
[[{'clicks': 4115}]]

ids: ['NegDHontRoulette3Clk002View00002OLin0802HLin1002']
[[{'clicks': 3653}]]

ids: ['NegDHontRoulette3Clk0005View00002OLin0802HLin1002']
[[{'clicks': 3582}]]

ids: ['NegDHontRoulette1Clk0005View00002OStat08HLin1002']
[[{'clicks': 2955}]]

ids: ['NegDHontRoulette3Clk02View00002OLin0802HLin1002']
[[{'clicks': 4104}]]

ids: ['NegDHontRoulette3Clk002View00001OLin0802HLin1002']
[[{'clicks': 3702}]]

ids: ['NegDHontRoulette3Clk002View00005OLin0802HLin1002']
[[{'clicks': 3602}]]

ids: ['NegDHontRoulette3Clk01View00005OLin0802HLin1002']
[[{'clicks': 4098}]]

ids: ['NegDHontRoulette3Clk0005View00005OLin0802HLin1002']
[[{'clicks': 3601}]]

ids: ['NegDHontRoulette1Clk0005View00005OStat08HLin1002']
[[{'clicks': 2923}]]

ids: ['NegDHontRoulette3Clk02View00005OLin0802HLin1002']
[[{'clicks': 4498}]]

ids: ['NegDHontRoulette1Clk0005View00005OLin0802HLin1002']
[[{'clicks': 2937}]]

ids: ['DHontFixedClk02View00001']
[[{'clicks': 3831}]]

ids: ['DHontFixedClk0005View00002']
[[{'clicks': 3671}]]

ids: ['DHontFixedClk01View00005']
[[{'clicks': 3933}]]

ids: ['DHontFixedClk0005View00005']
[[{'clicks': 3668}]]

ids: ['DHontFixedClk01View00002']
[[{'clicks': 3931}]]

ids: ['DHontFixedClk002View00002']
[[{'clicks': 3679}]]

ids: ['DHontFixedClk02View00005']
[[{'clicks': 3733}]]

ids: ['DHontFixedClk002View00001']
[[{'clicks': 3764}]]

ids: ['DHontFixedClk01View00001']
[[{'clicks': 3824}]]

ids: ['DHontFixedClk0005View00001']
[[{'clicks': 3669}]]

ids: ['DHontFixedClk002View00005']
[[{'clicks': 3669}]]

ids: ['DHontRoulette1Clk01View00002']
[[{'clicks': 4542}]]

ids: ['DHontRoulette3Clk0005View00005']
[[{'clicks': 3605}]]

ids: ['DHontRoulette1Clk002View00001']
[[{'clicks': 3015}]]

ids: ['DHontRoulette1Clk02View00001']
[[{'clicks': 4767}]]

ids: ['DHontRoulette1Clk0005View00001']
[[{'clicks': 2996}]]

ids: ['DHontRoulette1Clk02View00005']
[[{'clicks': 3668}]]

ids: ['DHontRoulette3Clk0005View00001']
[[{'clicks': 3596}]]

ids: ['DHontRoulette1Clk01View00001']
[[{'clicks': 4758}]]

ids: ['DHontRoulette1Clk01View00005']
[[{'clicks': 3237}]]

ids: ['DHontRoulette1Clk002View00002']
[[{'clicks': 3015}]]

ids: ['DHontRoulette3Clk002View00001']
[[{'clicks': 3716}]]

ids: ['DHontRoulette1Clk02View00002']
[[{'clicks': 4485}]]

ids: ['DHontRoulette3Clk0005View00002']
[[{'clicks': 3636}]]

ids: ['DHontRoulette1Clk0005View00002']
[[{'clicks': 2989}]]

ids: ['DHontRoulette3Clk002View00005']
[[{'clicks': 3591}]]

ids: ['DHontRoulette3Clk02View00005']
[[{'clicks': 4164}]]

ids: ['DHontRoulette1Clk0005View00005']
[[{'clicks': 2978}]]

ids: ['DHontRoulette3Clk01View00001']
[[{'clicks': 3819}]]

ids: ['DHontRoulette1Clk002View00005']
[[{'clicks': 2978}]]

ids: ['DHontRoulette3Clk002View00002']
[[{'clicks': 3660}]]

ids: ['DHontRoulette3Clk01View00002']
[[{'clicks': 3877}]]

ids: ['DHontRoulette3Clk02View00001']
[[{'clicks': 3820}]]

ids: ['DHontRoulette3Clk01View00005']
[[{'clicks': 4030}]]

ids: ['DHontRoulette3Clk02View00002']
[[{'clicks': 3849}]]

ids: ['NegDHontFixedClk02View00002OLin0802HLin1002']
[[{'clicks': 4495}]]

ids: ['NegDHontFixedClk01View00002OStat08HLin1002']
[[{'clicks': 4536}]]

ids: ['NegDHontFixedClk01View00005OLin0802HLin1002']
[[{'clicks': 4931}]]

ids: ['NegDHontFixedClk01View00001OStat08HLin1002']
[[{'clicks': 4510}]]

ids: ['NegDHontFixedClk0005View00002OLin0802HLin1002']
[[{'clicks': 4348}]]

ids: ['NegDHontFixedClk002View00005OStat08HLin1002']
[[{'clicks': 4304}]]

ids: ['NegDHontFixedClk02View00005OLin0802HLin1002']
[[{'clicks': 4571}]]

ids: ['NegDHontFixedClk02View00005OStat08HLin1002']
[[{'clicks': 4905}]]

ids: ['NegDHontFixedClk02View00002OStat08HLin1002']
[[{'clicks': 4572}]]

ids: ['NegDHontFixedClk0005View00001OLin0802HLin1002']
[[{'clicks': 4368}]]

ids: ['NegDHontFixedClk002View00005OLin0802HLin1002']
[[{'clicks': 4406}]]

ids: ['NegDHontFixedClk01View00002OLin0802HLin1002']
[[{'clicks': 4508}]]

ids: ['NegDHontFixedClk01View00001OLin0802HLin1002']
[[{'clicks': 4462}]]

ids: ['NegDHontFixedClk02View00001OLin0802HLin1002']
[[{'clicks': 4460}]]

ids: ['NegDHontFixedClk002View00001OStat08HLin1002']
[[{'clicks': 4574}]]

ids: ['NegDHontFixedClk002View00002OStat08HLin1002']
[[{'clicks': 4359}]]

ids: ['NegDHontFixedClk002View00001OLin0802HLin1002']
[[{'clicks': 4626}]]

ids: ['NegDHontFixedClk0005View00001OStat08HLin1002']
[[{'clicks': 4277}]]

ids: ['NegDHontFixedClk02View00001OStat08HLin1002']
[[{'clicks': 4519}]]

ids: ['NegDHontFixedClk002View00002OLin0802HLin1002']
[[{'clicks': 4415}]]

ids: ['NegDHontFixedClk0005View00005OLin0802HLin1002']
[[{'clicks': 4397}]]

ids: ['NegDHontFixedClk0005View00002OStat08HLin1002']
[[{'clicks': 4281}]]

ids: ['NegDHontFixedClk0005View00005OStat08HLin1002']
[[{'clicks': 4275}]]

ids: ['NegDHontRoulette1Clk02View00005OLin0802HLin1002']
[[{'clicks': 3612}]]

ids: ['NegDHontRoulette1Clk02View00002OLin0802HLin1002']
[[{'clicks': 4469}]]

ids: ['NegDHontRoulette3Clk01View00002OStat08HLin1002']
[[{'clicks': 4244}]]

ids: ['NegDHontRoulette3Clk0005View00005OStat08HLin1002']
[[{'clicks': 3568}]]

ids: ['NegDHontRoulette3Clk002View00001OStat08HLin1002']
[[{'clicks': 3661}]]

ids: ['NegDHontRoulette3Clk002View00002OStat08HLin1002']
[[{'clicks': 3568}]]

ids: ['NegDHontRoulette3Clk02View00005OStat08HLin1002']
[[{'clicks': 4817}]]

ids: ['NegDHontRoulette3Clk0005View00002OStat08HLin1002']
[[{'clicks': 3565}]]

ids: ['NegDHontRoulette3Clk002View00005OStat08HLin1002']
[[{'clicks': 3579}]]

ids: ['NegDHontRoulette1Clk02View00001OLin0802HLin1002']
[[{'clicks': 4787}]]

ids: ['NegDHontRoulette3Clk0005View00001OStat08HLin1002']
[[{'clicks': 3381}]]

ids: ['NegDHontRoulette3Clk02View00001OStat08HLin1002']
[[{'clicks': 4138}]]

ids: ['NegDHontRoulette3Clk01View00001OStat08HLin1002']
[[{'clicks': 4170}]]

ids: ['NegDHontRoulette3Clk0005View00001OLin0802HLin1002']
[[{'clicks': 3636}]]

ids: ['NegDHontRoulette3Clk01View00005OStat08HLin1002']
[[{'clicks': 4133}]]

ids: ['NegDHontRoulette3Clk02View00002OStat08HLin1002']
[[{'clicks': 4119}]]
"""


visualizationMatrix()