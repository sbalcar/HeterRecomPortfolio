#!/usr/bin/python3

import os

from typing import List
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription  # class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolDHondt import EvalToolDHondt  # class
from evaluationTool.evalToolHybrid import EToolHybrid  # class
from evaluationTool.evalToolDHondtPersonal import EvalToolDHondtPersonal #class

from aggregationDescription.aggregationDescription import AggregationDescription  # class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomMLDefinition import InputRecomMLDefinition  # class

from aggregation.operators.aDHondtSelector import ADHondtSelector  # class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector  # class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionML import ABatchDefinitionML  # class
from batchDefinition.ml1m.batchDefMLHybrid import BatchDefMLHybrid #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition  # class

from simulator.simulator import Simulator  # class

from history.historyHierDF import HistoryHierDF  # class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt  # class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalised import PModelDHondtPersonalised #class
from portfolioModel.pModelHybrid import PModelHybrid #class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class



class BatchDefMLHybridSkip(ABatchDefinitionML):

    mGlobalLrClicks:List[float] = BatchDefMLFuzzyDHondt.lrClicks
    mGlobalLrViewDivisors:List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    mGlobalNormOfRespons:List[bool] = [True, False]
    mPersonLrClicks:List[float] = BatchDefMLFuzzyDHondt.lrClicks
    mPersonLrViewDivisors:List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    mPersonNormOfRespons:List[bool] = [True, False]
    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs


    def getBatchName(self):
        return "HybridFDHondtSkip"

    def getParameters(self):
        batchDefMLHybrid = BatchDefMLHybrid()
        batchDefMLHybrid.mGlobalLrClicks = self.mGlobalLrClicks
        batchDefMLHybrid.mGlobalLrViewDivisors = self.mGlobalLrViewDivisors
        batchDefMLHybrid.mGlobalNormOfRespons = self.mGlobalNormOfRespons
        batchDefMLHybrid.mPersonLrClicks = self.mPersonLrClicks
        batchDefMLHybrid.mPersonLrViewDivisors = self.mPersonLrViewDivisors
        batchDefMLHybrid.mPersonNormOfRespons = self.mPersonNormOfRespons
        batchDefMLHybrid.selectorIDs = self.selectorIDs
        return batchDefMLHybrid.getParameters()


    def run(self, batchID: str, jobID: str):
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = \
            InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        # eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomMLDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondt(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        rIds:List[str] = pDescr.getRecommendersIDs()
        model:DataFrame = PModelHybrid(PModelDHondt(rIds), PModelDHondtPersonalisedStat(rIds),
                                       {PModelHybrid.ARG_MODE_SKIP:True})

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorML1M(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefMLHybridSkip().generateAllBatches(InputABatchDefinition())