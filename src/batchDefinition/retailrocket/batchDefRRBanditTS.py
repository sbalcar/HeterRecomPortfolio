#!/usr/bin/python3

import os
from typing import List

from pandas.core.frame import DataFrame  # class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription  # class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS  # class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class

from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition  # class

from batchDefinition.ml1m.batchDefMLBanditTS import BatchDefMLBanditTS  # class

from batchDefinition.inputABatchDefinition import InputABatchDefinition
from batchDefinition.aBatchDefinitionRR import ABatchDefinitionRR  # class

from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt  # class
from aggregation.operators.aDHondtSelector import ADHondtSelector  # class
from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition  # class
from batchDefinition.ml1m.batchDefMLFuzzyDHondt import BatchDefMLFuzzyDHondt #class

from simulator.simulator import Simulator  # class

from history.historyHierDF import HistoryHierDF  # class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


class BatchDefRRBanditTS(ABatchDefinitionRR):

    selectorIDs = [BatchDefMLFuzzyDHondt.SLCTR_FIXED, BatchDefMLFuzzyDHondt.SLCTR_ROULETTE1,
                   BatchDefMLFuzzyDHondt.SLCTR_ROULETTE2, BatchDefMLFuzzyDHondt.SLCTR_ROULETTE5]

    def getBatchName(self):
        return "BanditTS"

    def getParameters(self):
        batchDefMLBanditTS = BatchDefMLBanditTS()
        batchDefMLBanditTS.selectorIDs = self.selectorIDs
        return batchDefMLBanditTS.getParameters()

    def run(self, batchID: str, jobID: str):
        divisionDatasetPercentualSize:int
        uBehaviour:str
        repetition:int
        divisionDatasetPercentualSize, uBehaviour, repetition = \
            InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        selector:ADHondtSelector = self.getParameters()[jobID]

        rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()

        pDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            self.getBatchName() + jobID, rIDs, rDescs, InputAggrDefinition.exportADescBanditTS(selector))

        eTool:AEvalTool = EvalToolBanditTS({})
        model:DataFrame = PModelBandit(pDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition().exportSimulatorRetailRocket(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefRRBanditTS().generateAllBatches(InputABatchDefinition())
    #BatchDefRRBanditTS().run('rrDiv90Ulinear0109R1', 'Fixed')