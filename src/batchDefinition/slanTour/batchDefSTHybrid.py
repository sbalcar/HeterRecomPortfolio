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

from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition  # class

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
from portfolioModel.pModelHybrid import PModelHybrid #class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class



class BatchDefSTHybrid(ABatchDefinitionST):

    mGlobalLrClicks:List[float] = BatchDefMLFuzzyDHondt.lrClicks
    mGlobalLrViewDivisors:List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    mGlobalNormOfRespons:List[bool] = BatchDefMLFuzzyDHondt.normOfRespons
    mPersonLrClicks: List[float] = BatchDefMLFuzzyDHondt.lrClicks
    mPersonLrViewDivisors: List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    mPersonNormOfRespons:List[bool] = BatchDefMLFuzzyDHondt.normOfRespons
    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs


    def getBatchName(self):
        return "HybridFDHondt"

    def getParameters(self):
        aDict:Dict[str,object] = {}
        for selectorIDI in self.selectorIDs:
            for gLrClickJ in self.mGlobalLrClicks:
                for gLrViewDivisorK in self.mGlobalLrViewDivisors:
                    for gNormOfResponsL in self.mGlobalNormOfRespons:
                        for pLrClickM in self.mPersonLrClicks:
                            for pLrViewDivisorN in self.mPersonLrViewDivisors:
                                for pLrViewDivisorO in self.mPersonNormOfRespons:

                                    gjk:str = "Clk" + str(gLrClickJ).replace(".", "") + "ViewDivisor" + str(gLrViewDivisorK).replace(".", "") + "NR" + str(gNormOfResponsL)
                                    pjk:str = "Clk" + str(pLrClickM).replace(".", "") + "ViewDivisor" + str(pLrViewDivisorN).replace(".", "") + "NR" + str(pLrViewDivisorO)
                                    keyIJ:str = selectorIDI + gjk + pjk
                                    lrViewJK:float = gLrClickJ / gLrViewDivisorK
                                    lrViewLM:float = pLrClickM / pLrViewDivisorN
                                    evalToolMGlobal:EvalToolDHondt = EvalToolDHondt({
                                                EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: gLrClickJ,
                                                EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewJK,
                                                EvalToolDHondt.ARG_NORMALIZATION_OF_RESPONSIBILITY: gNormOfResponsL})
                                    evalToolMPerson:EvalToolDHondt = EvalToolDHondtPersonal({
                                                EvalToolDHondtPersonal.ARG_LEARNING_RATE_CLICKS: pLrClickM,
                                                EvalToolDHondtPersonal.ARG_LEARNING_RATE_VIEWS: lrViewLM,
                                                EvalToolDHondtPersonal.ARG_NORMALIZATION_OF_RESPONSIBILITY: pLrViewDivisorO})
                                    eToolIJK:AEvalTool = EToolHybrid(evalToolMGlobal, evalToolMPerson, {})
                                    selectorIJK:ADHondtSelector = BatchDefMLFuzzyDHondt().getSelectorParameters()[selectorIDI]
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

        rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()

        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondt(selector)

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, aDescDHont)

        rIds:List[str] = pDescr.getRecommendersIDs()
        model:DataFrame = PModelHybrid(PModelDHondt(rIds), PModelDHondtPersonalised(rIds))

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorSlantour(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefSTHybrid().generateAllBatches(InputABatchDefinition())