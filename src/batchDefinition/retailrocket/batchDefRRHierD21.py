#!/usr/bin/python3

import os
from typing import List

from pandas.core.frame import DataFrame  # class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription  # class
from portfolioDescription.portfolioHierDescription import PortfolioHierDescription #class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolBanditTS import EvalToolBanditTS  # class
from evaluationTool.evalToolDoNothing import EToolDoNothing  # class

from batchDefinition.inputAggrDefinition import InputAggrDefinition  # class
from batchDefinition.modelDefinition import ModelDefinition

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

from aggregationDescription.aggregationDescription import AggregationDescription #class
from recommenderDescription.recommenderDescription import RecommenderDescription #class

from aggregation.negImplFeedback.aPenalization import APenalization #class
from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.aggrD21 import AggrD21 #class
from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
from aggregation.negImplFeedback.penalUsingProbability import PenalUsingProbability #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class


class BatchDefRRHierD21(ABatchDefinitionRR):

    selectorIDs:List[str] = [BatchDefMLFuzzyDHondt.SLCTR_FIXED]

    def getBatchName(self):
        return "HierD21"

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

        recommenderID:str = "TheMostPopular"
        pRDescr: RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})

        selector:ADHondtSelector = self.getParameters()[jobID]
        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimizeThompsonSampling(
            selector)
        aDescDHont:AggregationDescription = InputAggrDefinition.exportADescFAI()

        rIDs:List[str]
        rDescs:List[AggregationDescription]
        rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()
        #rIDs = [recommenderID]
        #rDescs = [pRDescr]

        p1AggrDescrID:str = "p1AggrDescrID"
        p1AggrDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(p1AggrDescrID, rIDs, rDescs, aDescDHont)


        #pProbTool:APenalization = PenalizationToolDefinition.exportProbPenaltyToolOLin0802HLin1002(
        #    InputSimulatorDefinition.numberOfAggrItems)
        pProbTool:APenalization = PenalizationToolDefinition.exportPenaltyToolOStat08HLin1002(
            InputSimulatorDefinition.numberOfAggrItems)

        aHierDescr:AggregationDescription = AggregationDescription(AggrD21,
                                    {AggrD21.ARG_RATING_THRESHOLD_FOR_NEG: 0.0})

        pHierDescr:PortfolioHierDescription = PortfolioHierDescription("pHierDescr",
                                        recommenderID, pRDescr, p1AggrDescrID,
                                        p1AggrDescr,
                                        aHierDescr,
                                        pProbTool)

        #eTool:AEvalTool = EvalToolBanditTS({})
        eTool:AEvalTool = EToolDoNothing({})
        #model:DataFrame = ModelDefinition.createDHontModel(p1AggrDescr.getRecommendersIDs())
        model:DataFrame = ModelDefinition.createBanditModel(p1AggrDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorRetailRocket(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pHierDescr], [model], [eTool], [HistoryHierDF(p1AggrDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    #BatchDefRRHierD21().generateAllBatches(InputABatchDefinition())
    BatchDefRRHierD21().run('rrDiv90Ulinear0109R1', 'Fixed')