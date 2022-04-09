#!/usr/bin/python3

import os
import numpy as np

from typing import List #class
from typing import Dict #class

from pandas.core.frame import DataFrame #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolDHondt import EvalToolDHondt #class
from evaluationTool.evalToolContext import EvalToolContext #class
from evaluationTool.evalToolDHondtPersonal import EvalToolDHondtPersonal #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  #class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.ml1m.batchDefMLBanditTS import BatchDefMLBanditTS #class
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

from simulator.simulator import Simulator #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class
from history.historyHierDF import HistoryHierDF #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class

from batchDefinition.ml1m.batchDefMLFuzzyDHondtDirectOptimize import BatchDefMLFuzzyDHondtDirectOptimize #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalised import PModelDHondtPersonalised #class


class BatchDefMLCluster(ABatchDefinitionML):

    lrClicks:List[float] = [0.2, 0.1, 0.03, 0.005]
    lrViewDivisors:List[float] = [250, 500, 1000]

    selectorIDs = [BatchDefMLFuzzyDHondtDirectOptimize.SLCTR_FIXED]

    def getBatchName(self):
        return "Cluster"

    def getParameters(self):
        aDict:Dict[str,object] = {}
        for selectorIDI in self.selectorIDs:
            for lrClickJ in self.lrClicks:
                for lrViewDivisorK in self.lrViewDivisors:

                    normOfResponsL:bool = False

                    keyIJ:str = selectorIDI + "Clk" + str(lrClickJ).replace(".", "") + "ViewDivisor" + str(lrViewDivisorK).replace(".", "") + "NR" + str(normOfResponsL)
                    lrViewIJK:float = lrClickJ / lrViewDivisorK
                    eToolIJK:AEvalTool = EvalToolDHondtPersonal({
                                    EvalToolDHondtPersonal.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                    EvalToolDHondtPersonal.ARG_LEARNING_RATE_VIEWS: lrViewIJK,
                                    EvalToolDHondtPersonal.ARG_NORMALIZATION_OF_RESPONSIBILITY: normOfResponsL})
                    selectorIJK:ADHondtSelector = BatchDefMLFuzzyDHondt().getAllSelectors()[selectorIDI]
                    aDict[keyIJ] = (selectorIJK, eToolIJK)
        return aDict

    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        portfolioID:str = self.getBatchName() + jobID

        history:AHistory = HistoryHierDF(portfolioID)

        #eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrsCluster()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondt(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        print(pDescr.getRecommendersIDs())
        model:DataFrame = PModelDHondtPersonalised(pDescr.getRecommendersIDs())

        inputSimulatorDefinition = InputSimulatorDefinition()
        inputSimulatorDefinition.numberOfAggrItems = 100
        simulator:Simulator = inputSimulatorDefinition.exportSimulatorML1M(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [history])



if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    #BatchDefMLCluster.generateAllBatches(InputABatchDefinition())


    #file:str = "results/ml1mDiv90Ustatic06R1/portfModelTimeEvolution-ClusterFixedClk003ViewDivisor500NRFalse.txt"
    file:str = "../results/ml1mDiv90Ulinear0109R1/portfModelTimeEvolution-ClusterFixedClk003ViewDivisor500NRFalse.txt"
    modelDF:PModelDHondtPersonalised = PModelDHondtPersonalised.readModel(file, 38000)
    #print(modelDF.head())

    userID = 11.0

    modelOfUserI:DataFrame = modelDF.getModel(float('nan'))
    modelOfUserI:DataFrame = modelDF.getModel(userID)

    print(modelOfUserI.head(25))
    print(list(modelDF.index))

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()
    #print(ratingsDF.head())


    ratingsOfuser:DataFrame = ratingsDF[ratingsDF[Ratings.COL_USERID] == userID]
    itemIds:List[int] = ratingsOfuser[Ratings.COL_MOVIEID].tolist()
    #print(itemIds)

    itemsDF:DataFrame = Items.readFromFileMl1m()
    r = Items.countA(itemsDF, itemIds)
    print(r)


    #from matplotlib import pyplot as plt
    #from matplotlib import font_manager as fm

    # make a square figure and axes
    #fig = plt.figure(1, figsize=(6, 6), dpi=50)
    #ax = fig.add_axes([0.16, 0.16, 0.68, 0.68])

    #plt.title("Scripting languages")
    #ax.title.set_fontsize(30)

    # vytvoření koláčového grafu
    #ax.pie(r.values(), labels=r.keys(), autopct='%1.1f%%', shadow=True)
    #ax.pie(r.values(), labels=r.keys(), autopct='%1.1f%%', shadow=True)
    #plt.show()

    for userIdI in modelDF.index:
        print("userIdI: " + str(userIdI))
        if np.isnan(userIdI):
            continue

        modelOfUserI:DataFrame = modelDF.getModel(userIdI)

        ratingsOfuser:DataFrame = ratingsDF[ratingsDF[Ratings.COL_USERID] == userIdI]
        itemIds:List[int] = ratingsOfuser[Ratings.COL_MOVIEID].tolist()
        r = Items.countA(itemsDF, itemIds)

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        import numpy as np
        np.random.seed(123456)
        import pandas as pd
        #df = pd.DataFrame(np.random.rand(4, 2), index=Items.getAllGenres()[:4], columns=['UserProfileAnalysis', 'DHondtModel'])
        #df = pd.DataFrame(zip([0.1,0.2,0.3,0.4,0.1,0.2], [0.1,0.2,0.3,0.4,0.1,0.2]), index=['a','b','c','d','e','f'], columns=['UserProfileAnalysis', 'DHondtModel'])
        #df = pd.DataFrame(zip(r.values(), r.values()), index=r.keys(), columns=['UserProfileAnalysis', 'DHondtModel'])
        df = pd.DataFrame(zip(r.values(), modelOfUserI['votes'].tolist()), index=r.keys(), columns=['UserProfileAnalysis', 'DHondtModel'])

        f, axes = plt.subplots(1,2, figsize=(10,5))
        for ax, col in zip(axes, df.columns):
            df[col].plot(kind='pie', autopct='%.2f', ax=ax, title=col, fontsize=10)
            ax.legend(loc=3)
            plt.ylabel("")
            plt.xlabel("")

        #plt.show()
        plt.savefig('../graphs/user-'+ str(userIdI) + '.png')