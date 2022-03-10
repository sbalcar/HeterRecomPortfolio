#!/usr/bin/python3

import os

from typing import List # class
from typing import Dict  # class

from pandas.core.frame import DataFrame  # class

from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription  # class

from evaluationTool.aEvalTool import AEvalTool  # class
from evaluationTool.evalToolDHondt import EvalToolDHondt  # class
from evaluationTool.evalToolDHondtPersonal import EvalToolDHondtPersonal #class

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
from portfolioModel.pModelDHondtPersonalised import PModelDHondtPersonalised #class
from portfolioModel.pModelDHondtPersonalisedStat import PModelDHondtPersonalisedStat #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from portfolioDescription.aPortfolioDescription import APortfolioDescription #class
from portfolioDescription.portfolioDynamicDescription import PortfolioDynamicDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class


class BatchDefRRDynamic(ABatchDefinitionRR):

    lrClicks:List[float] = BatchDefMLFuzzyDHondt.lrClicks
    lrViewDivisors:List[float] = BatchDefMLFuzzyDHondt.lrViewDivisors
    selectorIDs:List[str] = BatchDefMLFuzzyDHondt.selectorIDs


    def getBatchName(self):
        return "DynamicFDHondtPersStat"


    def getParameters(self):
        aDict:Dict[str,object] = {}
        for selectorIDI in self.selectorIDs:
            for lrClickJ in self.lrClicks:
                for lrViewDivisorK in self.lrViewDivisors:
                    keyIJ:str = selectorIDI + "Clk" + str(lrClickJ).replace(".", "") + "ViewDivisor" + str(lrViewDivisorK).replace(".", "")
                    lrViewIJK:float = lrClickJ / lrViewDivisorK
                    eToolIJK:AEvalTool = EvalToolDHondtPersonal({EvalToolDHondt.ARG_LEARNING_RATE_CLICKS: lrClickJ,
                                                                 EvalToolDHondt.ARG_LEARNING_RATE_VIEWS: lrViewIJK})
                    selectorIJK:ADHondtSelector = BatchDefMLFuzzyDHondt().getAllSelectors()[selectorIDI]
                    aDict[keyIJ] = (selectorIJK, eToolIJK)
        return aDict


    def run(self, batchID: str, jobID: str):
        divisionDatasetPercentualSize: int
        uBehaviour: str
        repetition: int
        divisionDatasetPercentualSize, uBehaviour, repetition = \
            InputABatchDefinition().getBatchParameters(self.datasetID)[batchID]

        # eTool:AEvalTool
        selector, eTool = self.getParameters()[jobID]


        rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()

        p1AggrDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(
            "FDHont" + jobID, rIDs, rDescs, InputAggrDefinition.exportADescDHondt(selector))

        recommenderID: str = "TheMostPopular"
        rDescr:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})

        pDescr:APortfolioDescription = PortfolioDynamicDescription(
            "Dynamic" + "FDHontPersStat" + jobID, recommenderID, rDescr, "FDHondt", p1AggrDescr)

        model:DataFrame = PModelDHondtPersonalisedStat(p1AggrDescr.getRecommendersIDs())

        simulator:Simulator = InputSimulatorDefinition.exportSimulatorRetailRocket(
            batchID, divisionDatasetPercentualSize, uBehaviour, repetition)
        simulator.simulate([pDescr], [model], [eTool], [HistoryHierDF(pDescr.getPortfolioID())])


if __name__ == "__main__":
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())

    BatchDefRRDynamic().generateAllBatches(InputABatchDefinition())