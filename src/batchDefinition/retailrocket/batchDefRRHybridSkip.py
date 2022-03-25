#!/usr/bin/python3

import os

from typing import List
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription  # class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolDHondt import EvalToolDHondt  # class
from evaluationTool.evalToolHybrid import EToolHybrid  # class
from evaluationTool.evalToolDHondtPersonal import EvalToolDHondtPersonal  # class

from aggregationDescription.aggregationDescription import AggregationDescription  # class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition  # class

from aggregation.operators.aDHondtSelector import ADHondtSelector  # class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector  # class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionRR import ABatchDefinitionRR  # class
from batchDefinition.retailrocket.batchDefRRHybrid import BatchDefRRHybrid # class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition  # class

from simulator.simulator import Simulator  # class

from history.historyHierDF import HistoryHierDF  # class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt  # class

from portfolioModel.pModelBandit import PModelBandit  # class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes  # class
from portfolioModel.pModelDHondt import PModelDHondt  # class
from portfolioModel.pModelDHondtPersonalised import PModelDHondtPersonalised  # class
from portfolioModel.pModelHybrid import PModelHybrid  # class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat  # class


class BatchDefRRHybridSkip(ABatchDefinitionRR):

    mGlobalLrClicks:List[float] = BatchDefRRHybrid.mGlobalLrClicks
    mGlobalLrViewDivisors:List[float] = BatchDefRRHybrid.mGlobalLrViewDivisors
    mGlobalNormOfRespons:List[bool] = BatchDefRRHybrid.mGlobalNormOfRespons
    mPersonLrClicks:List[float] = BatchDefRRHybrid.mPersonLrClicks
    mPersonLrViewDivisors:List[float] = BatchDefRRHybrid.mPersonLrViewDivisors
    mPersonNormOfRespons:List[bool] = BatchDefRRHybrid.mPersonNormOfRespons
    selectorIDs:List[str] = BatchDefRRHybrid.selectorIDs


    def getBatchName(self):
        return "HybridFDHondtSkip"

    def getParameters(self):
        batchDefRRHybrid = BatchDefRRHybrid()
        batchDefRRHybrid.mGlobalLrClicks = self.mGlobalLrClicks
        batchDefRRHybrid.mGlobalLrViewDivisors = self.mGlobalLrViewDivisors
        batchDefRRHybrid.mGlobalNormOfRespons = self.mGlobalNormOfRespons
        batchDefRRHybrid.mPersonLrClicks = self.mPersonLrClicks
        batchDefRRHybrid.mPersonLrViewDivisors = self.mPersonLrViewDivisors
        batchDefRRHybrid.mPersonNormOfRespons = self.mPersonNormOfRespons
        batchDefRRHybrid.selectorIDs = self.selectorIDs
        return batchDefRRHybrid.getParameters()

    def run(self, batchID:str, jobID:str):
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = \
            InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        # eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondt(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        rIds:List[str] = pDescr.getRecommendersIDs()
        model:DataFrame = PModelHybrid(PModelDHondt(rIds), PModelDHondtPersonalisedStat(rIds),
                                       {PModelHybrid.ARG_MODE_SKIP:True})

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorRetailRocket(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefRRHybrid().generateAllBatches(InputABatchDefinition())