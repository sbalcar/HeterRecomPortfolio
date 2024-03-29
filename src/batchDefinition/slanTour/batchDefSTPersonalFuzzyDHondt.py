#!/usr/bin/python3

import os

from typing import List
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription  # class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolDHondt import EvalToolDHondt  # class
from evaluationTool.evalToolDHondtPersonal import EvalToolDHondtPersonal #class

from aggregationDescription.aggregationDescription import AggregationDescription  # class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomSTDefinition import InputRecomSTDefinition  # class

from aggregation.operators.aDHondtSelector import ADHondtSelector  # class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector  # class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector  # class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionST import ABatchDefinitionST  # class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition  # class

from simulator.simulator import Simulator  # class

from history.historyHierDF import HistoryHierDF  # class

from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt  # class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class
from portfolioModel.pModelDHondtPersonalised import PModelDHondtPersonalised #class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class


class BatchDefSTPersonalFuzzyDHondt(ABatchDefinitionST):

    lrClicks:List[float] = BatchDefMLFuzzyDHondt.lrClicks
    lrViewDivisors:List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    normOfRespons:List[bool] = BatchDefMLFuzzyDHondt.normOfRespons
    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs


    def getBatchName(self):
        return "PersonalFDHondt"


    def getParameters(self):
        aDict:Dict[str,object] = {}
        for selectorIDI in self.selectorIDs:
            for lrClickJ in self.lrClicks:
                for lrViewDivisorK in self.lrViewDivisors:
                   for normOfResponsL in self.normOfRespons:

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
        divisionDatasetPercentualSize, uBehaviour, repetition = \
            InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        # eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomSTDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondt(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        model:DataFrame = PModelDHondtPersonalised(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition().exportSimulatorSlantour(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefSTPersonalFuzzyDHondt().generateAllBatches(InputABatchDefinition())