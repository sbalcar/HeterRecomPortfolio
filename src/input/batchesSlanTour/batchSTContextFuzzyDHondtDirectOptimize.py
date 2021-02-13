#!/usr/bin/python3

import os
from typing import List #class
from typing import Dict #class

from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.items import Items #class
from datasets.ml.users import Users #class
from datasets.ml.behavioursML import BehavioursML #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class

from pandas.core.frame import DataFrame #class

from simulator.simulator import Simulator #class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from evaluationTool.aEvalTool import AEvalTool #class
from evaluationTool.evalToolContext import EvalToolContext #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from input.inputAggrDefinition import InputAggrDefinition  # class
from input.modelDefinition import ModelDefinition

from input.inputRecomSTDefinition import InputRecomSTDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class

from aggregation.negImplFeedback.aPenalization import APenalization #class

from input.inputABatchDefinition import InputABatchDefinition
from input.aBatchST import ABatchST #class
from input.batchesML1m.batchMLFuzzyDHondtThompsonSamplingINF import BatchMLFuzzyDHondtThompsonSamplingINF #class

from input.inputSimulatorDefinition import InputSimulatorDefinition #class

from simulator.simulator import Simulator #class

from history.historyHierDF import HistoryHierDF #class
from input.batchesML1m.batchMLFuzzyDHondt import BatchMLFuzzyDHondt #class
from input.batchesML1m.batchMLContextFuzzyDHondtDirectOptimize import BatchMLContextFuzzyDHondtDirectOptimize #class

from history.aHistory import AHistory #class


class BatchSTContextFuzzyDHondtDirectOptimize(ABatchST):

    selectorIDs:List[str] = BatchMLFuzzyDHondt.selectorIDs

    @classmethod
    def getParameters(cls):

        aDict:Dict[str,object] = BatchMLFuzzyDHondt.getAllSelectors()
        aSubDict:Dict[str,object] = {selIdI: aDict[selIdI] for selIdI in aDict.keys() if selIdI in cls.selectorIDs}
        return aSubDict


    def run(self, batchID:str, jobID:str):

        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = InputABatchDefinition.getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector = self.getParameters()[jobID]

        dataset:ADataset = DatasetST.readDatasets()
        events = dataset.eventsDF
        serials = dataset.serialsDF

        historyDF:AHistory = HistoryHierDF("test01")

        # Init evalTool
        evalTool:AEvalTool = EvalToolContext({
            EvalToolContext.ARG_ITEMS: serials,  # ITEMS
            EvalToolContext.ARG_EVENTS: events,  # EVENTS (FOR CALCULATING HISTORY OF USER)
            EvalToolContext.ARG_DATASET: "st",  # WHAT DATASET ARE WE IN
            EvalToolContext.ARG_HISTORY: historyDF})  # empty instance of AHistory is OK for ST dataset

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescContextFuzzyDHondtDirectOptimize(selector, evalTool)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "ContextFDHondtDirectOptimize" + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = ModelDefinition.createDHontModel(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
                batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [evalTool], [HistoryHierDF(pDescr.getPortfolioID())])




if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchSTContextFuzzyDHondtDirectOptimize.generateAllBatches()
    #BatchSTContextFuzzyDHondtDirectOptimize().run("stDiv90Ulinear0109R1", "Fixed")
