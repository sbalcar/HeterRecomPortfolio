#!/usr/bin/python3

import os
from typing import List
from typing import Dict #class

#from pandas.core.series import Series #class

#from aggregation.negImplFeedback.penalUsingFiltering import PenalUsingFiltering #class
#from aggregation.negImplFeedback.aPenalization import APenalization #class
from pandas import DataFrame

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.aDataset import ADataset #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from portfolio.aPortfolio import APortfolio #class

from portfolioDescription.portfolio1MethDescription import Portfolio1MethDescription #class
from portfolioDescription.portfolioHierDescription import PortfolioHierDescription #class
from portfolioDescription.portfolio1AggrDescription import Portfolio1AggrDescription #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from batchDefinition.inputAggrDefinition import InputAggrDefinition #class

from aggregationDescription.aggregationDescription import AggregationDescription #class
from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition  # class

from aggregation.operators.aDHondtSelector import ADHondtSelector #class
from aggregation.operators.rouletteWheelSelector import RouletteWheelSelector #class
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class


from aggregation.negImplFeedback.aPenalization import APenalization #class
from batchDefinition.inputAggrDefinition import PenalizationToolDefinition #class

from batchDefinition.inputSimulatorDefinition import InputSimulatorDefinition #class

from aggregationDescription.aggregationDescription import AggregationDescription #class

from aggregation.aggrD21 import AggrD21 #class

from portfolioModel.pModelBandit import PModelBandit #class
from portfolioModel.pModelDHondtBanditsVotes import PModelDHondtBanditsVotes #class
from portfolioModel.pModelDHondt import PModelDHondt #class


def test01():
    print("Test 01")

    recommenderID:str = "TheMostPopular"
    pRDescr:RecommenderDescription = RecommenderDescription(RecommenderTheMostPopular, {})

    selectorFixed:ADHondtSelector = TheMostVotedItemSelector({})
    aDescDHont:AggregationDescription = InputAggrDefinition.exportADescDHondtDirectOptimizeThompsonSampling(selectorFixed)

    rIDs:List[str]
    rDescs:List[AggregationDescription]
    rIDs, rDescs = InputRecomRRDefinition.exportPairOfRecomIdsAndRecomDescrs()
    rIDs = [recommenderID]
    rDescs = [pRDescr]

    p1AggrDescrID:str = "p1AggrDescrID"
    p1AggrDescr:Portfolio1AggrDescription = Portfolio1AggrDescription(p1AggrDescrID, rIDs, rDescs, aDescDHont)

    pProbTool:APenalization = PenalizationToolDefinition.exportProbPenaltyToolOLin0802HLin1002(
        InputSimulatorDefinition.numberOfAggrItems)
    pProbTool:APenalization = PenalizationToolDefinition.exportPenaltyToolOStat08HLin1002(
        InputSimulatorDefinition.numberOfAggrItems)

    aHierDescr:AggregationDescription = AggregationDescription(AggrD21, {AggrD21.ARG_RATING_THRESHOLD_FOR_NEG:2.0})

    pHierDescr:PortfolioHierDescription = PortfolioHierDescription("pHierDescr",
                            recommenderID, pRDescr, p1AggrDescrID, p1AggrDescr,
                            aHierDescr,
                                                                   pProbTool)

    userID:int = 1

    dataset:ADataset = DatasetRetailRocket.readDatasetsWithFilter(minEventCount=50)

    history:AHistory = HistoryDF("test")
    history.insertRecommendation(userID, 45, 1,  False)
    history.insertRecommendation(userID, 45, 2,  False)
    history.insertRecommendation(userID, 78, 3, False)

    p:APortfolio = pHierDescr.exportPortfolio("test", history)

    portFolioModel:DataFrame = PModelDHondtBanditsVotes(p1AggrDescr.getRecommendersIDs())


    p.train(history, dataset)

    #df:DataFrame = DataFrame([[1, 555]], columns=[Events.COL_USER_ID, Events.COL_OBJECT_ID])
    #p.update(ARecommender.UPDT_CLICK, df)

    args = {APortfolio.ARG_NUMBER_OF_AGGR_ITEMS: 20,
                APortfolio.ARG_ITEM_ID:1,
                APortfolio.ARG_NUMBER_OF_RECOMM_ITEMS:100,
                AggrD21.ARG_RATING_THRESHOLD_FOR_NEG:0.5}

    r, rp = p.recommend(userID, portFolioModel, args)
    print(r)



if __name__ == "__main__":
    os.chdir("..")

    test01()