#!/usr/bin/python3

from typing import List

import os

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

import pandas as pd

from evaluationTool.evalToolContext import EvalToolContext #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetST import DatasetST #class


from recommender.aRecommender import ARecommender #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

from datasets.ml.users import Users #class
from datasets.ml.items import Items #class
from datasets.ml.ratings import Ratings #class

from aggregation.aggrContextFuzzyDHondtINF import AggrContextFuzzyDHondtINF #class
from aggregation.aggrFuzzyDHondt import AggrFuzzyDHondt
from aggregation.operators.theMostVotedItemSelector import TheMostVotedItemSelector #class


from aggregation.negImplFeedback.penalUsingReduceRelevance import penaltyLinear #function

from aggregation.negImplFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.negImplFeedback.aPenalization import APenalization #class


def test03():
    # First get Dataset Data
    dataset:ADataset = DatasetST.readDatasets()
    events = dataset.eventsDF
    serials = dataset.serialsDF

    # I created some dummy data, but each key,value pair should be result list from a recommender
    #   (=what recommender recommended)
    methodsResultDict: dict = {
        "metoda1": pd.Series([0.2, 0.1, 0.3, 0.3, 0.1], [32, 2, 8, 1, 4], name="rating"),
        "metoda2": pd.Series([0.1, 0.1, 0.2, 0.3, 0.3], [1, 5, 32, 6, 7], name="rating")
    }

    # init votes for each recommender
    portfolioModelData = [['metoda1', 0.6], ['metoda2', 0.4]]
    portfolioModelDF:DataFrame = pd.DataFrame(portfolioModelData, columns=["methodID","votes"])
    portfolioModelDF.set_index("methodID", inplace=True)


    userID = 1
    itemID = 20
    historyDF: AHistory = HistoryDF("test01")

    # WHAT EVALUATIOR NEEDS into dictionary!
    evaluationDict: dict = {EvalToolContext.ARG_USER_ID: userID,
                            EvalToolContext.ARG_ITEM_ID: itemID,    # ITEMID (not mandatory if EvalToolContext.ARG_PAGE_TYPE != "zobrazit")
                            EvalToolContext.ARG_SENIORITY: 5,   # SENIORITY OF USER
                            EvalToolContext.ARG_PAGE_TYPE: "zobrazit",  #   TYPE OF PAGE ("zobrazit", "index" or "katalog)
                            EvalToolContext.ARG_ITEMS_SHOWN: 10 # HOW MANY ITEMS ARE SHOWN TO USER
                            }
    # Init eTool
    eToolContext = EvalToolContext(
        {
         EvalToolContext.ARG_ITEMS: serials,    # ITEMS
         EvalToolContext.ARG_EVENTS: events,    # EVENTS (FOR CALCULATING HISTORY OF USER)
         EvalToolContext.ARG_DATASET: "st",     # WHAT DATASET ARE WE IN
         EvalToolContext.ARG_HISTORY: historyDF} # empty instance of AHistory is OK for ST dataset
    )
    penalization: APenalization = PenalUsingReduceRelevance(penaltyLinear, [0.8, 0.2, 3], penaltyLinear, [1.0, 0.2, 3],3)
    aggr: AggrContextFuzzyDHondtINF = AggrContextFuzzyDHondtINF(historyDF, {  # empty instance of AHistory is OK for ST dataset
        AggrContextFuzzyDHondtINF.ARG_EVAL_TOOL: eToolContext, # eTool
        AggrContextFuzzyDHondtINF.ARG_SELECTOR: TheMostVotedItemSelector({}), # ? FuzzyDHondt needs this, not contextAggr
        AggrContextFuzzyDHondtINF.ARG_PENALTY_TOOL:penalization
    })

    # Get data from aggregator
    rItemsWithResponsibility = aggr.runWithResponsibility(methodsResultDict, portfolioModelDF, userID, numberOfItems=5, argumentsDict=evaluationDict)
    # call click & displayed methods
    l1 = eToolContext.displayed(rItemsWithResponsibility, portfolioModelDF, evaluationDict)
    # rItemsWithResponsibility[0][0] is clicked item
    l1 = eToolContext.click(rItemsWithResponsibility,rItemsWithResponsibility[0][0],portfolioModelDF,evaluationDict)

    # ...
    # ...
    # ...
    # user is now on "index" page type, so we have to change page type in evaluationDict (!)
    evaluationDict[EvalToolContext.ARG_PAGE_TYPE] = "index"

    # same as before
    # Get data from aggregator
    rItemsWithResponsibility = aggr.runWithResponsibility(methodsResultDict, portfolioModelDF, userID, numberOfItems=5,
                                                          argumentsDict=evaluationDict)
    # call click & displayed methods
    l1 = eToolContext.displayed(rItemsWithResponsibility, portfolioModelDF, evaluationDict)
    # rItemsWithResponsibility[0][0] is clicked item
    l1 = eToolContext.click(rItemsWithResponsibility, rItemsWithResponsibility[0][0], portfolioModelDF, evaluationDict)


if __name__ == "__main__":
    os.chdir("..")
    test03()