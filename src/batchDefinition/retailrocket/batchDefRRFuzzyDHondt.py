#!/usr/bin/python3

import os

from typing import List
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription  # class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolDHondt import EvalToolDHondt  # class

from aggregationDescription.aggregationDescription import AggregationDescription  # class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition  # class

from aggregation.operators.aDHondtSelector import ADHondtSelector  # class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector  # class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionRR import ABatchDefinitionRR  # class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition  # class

from simulator.simulator import Simulator  # class

from history.historyHierDF import HistoryHierDF  # class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt  # class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefRRFuzzyDHondt(ABatchDefinitionRR):

    lrClicks:List[float] = BatchDefMLFuzzyDHondt.lrClicks
    lrViewDivisors:List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    normOfRespons:List[bool] = BatchDefMLFuzzyDHondt.normOfRespons
    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs

    def getBatchName(self):
        return "FDHondt"

    def getParameters(self):
        batchDefMLFuzzyDHondt = BatchDefMLFuzzyDHondt()
        batchDefMLFuzzyDHondt.lrClicks:List[float] = self.lrClicks
        batchDefMLFuzzyDHondt.lrViewDivisors:List[float] = self.lrViewDivisors
        batchDefMLFuzzyDHondt.normOfRespons:List[bool] = self.normOfRespons
        batchDefMLFuzzyDHondt.selectorIDs:List[str] = self.selectorIDs
        return batchDefMLFuzzyDHondt.getParameters()

    def run(self, batchID: str, jobID: str):
        divisionDatasetPercentualSize: int
        uBehaviour: str
        repetition: int
        divisionDatasetPercentualSize, uBehaviour, repetition = \
            InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        # eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondt(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = PModelDHondt(pDescr.getRecommendersIDs())

        simulator: Simulator = InputSimulatorDefinition.exportSimulatorRetailRocket(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefRRFuzzyDHondt().generateAllBatches(InputABatchDefinition())