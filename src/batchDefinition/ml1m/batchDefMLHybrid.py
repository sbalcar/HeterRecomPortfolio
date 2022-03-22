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



class BatchDefMLHybrid(ABatchDefinitionML):

    mGlobalLrClicks:List[float] = BatchDefMLFuzzyDHondt.lrClicks
    mGlobalLrViewDivisors:List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    mPersonLrClicks: List[float] = BatchDefMLFuzzyDHondt.lrClicks
    mPersonLrViewDivisors: List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs


    def getBatchName(self):
        return "HybridFDHondt"

    def getParameters(self):
        aDict:Dict[str,object] = {}
        for selectorIDI in self.selectorIDs:
            for gLrClickJ in self.mGlobalLrClicks:
                for gLrViewDivisorK in self.mGlobalLrViewDivisors:
                    for pLrClickL in self.mPersonLrClicks:
                        for pLrViewDivisorM in self.mPersonLrViewDivisors:

                            gjk:str = "Clk" + str(gLrClickJ).replace(".", "") + "ViewDivisor" + str(gLrViewDivisorK).replace(".", "")
                            pjk:str = "Clk" + str(pLrClickL).replace(".", "") + "ViewDivisor" + str(pLrViewDivisorM).replace(".", "")
                            keyIJ:str = selectorIDI + gjk + pjk
                            lrViewJK:float = gLrClickJ / gLrViewDivisorK
                            lrViewLM:float = pLrClickL / pLrViewDivisorM
                            evalToolMGlobal:EvalToolDHondt = EvalToolDHondt({
                                        EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: gLrClickJ,
                                        EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewJK})
                            evalToolMPerson:EvalToolDHondt = EvalToolDHondtPersonal({
                                        EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: pLrClickL,
                                        EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewLM})
                            eToolIJK:AEvalTool = EToolHybrid(evalToolMGlobal, evalToolMPerson, {})
                            selectorIJK:ADHondtSelector = BatchDefMLFuzzyDHondt().getSelectorParameters()[selectorIDI]
                            aDict[keyIJ] = (selectorIJK, eToolIJK)
        return aDict


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
        model:DataFrame = PModelHybrid(PModelDHondt(rIds), PModelDHondtPersonalisedStat(rIds))

        simulator: Simulator = InputSimulatorDefinition.exportSimulatorML1M(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefMLHybrid().generateAllBatches(InputABatchDefinition())