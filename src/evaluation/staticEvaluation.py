#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

from typing import List

import os
from pandas.core.frame import DataFrame #class

from datasets.aDataset import ADataset #class
from datasets.datasetRetailRocket import DatasetRetailRocket #class

from datasets.retailrocket.events import Events #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderBPRMFImplicit import RecommenderBPRMFImplicit #class
from recommender.recommenderVSKNN import RecommenderVMContextKNN #class

from batchDefinition.inputRecomRRDefinition import InputRecomRRDefinition #class
from recommenderDescription.recommenderDescription import RecommenderDescription #class

from history.aHistory import AHistory #class
from history.historyDF import HistoryDF #class

class StaticEvaluation:

    def __init__(self, dataset:ADataset):
        if not isinstance(dataset, ADataset):
            raise ValueError("Argument dataset isn't type ADataset.")

        self.dataset = dataset

    def evaluate(self, rDscr:RecommenderDescription):
        print("StaticEvaluation")

        recom:ARecommender = rDscr.exportRecommender("test")
        args:dict = rDscr.getArguments()

        eventsDF:DataFrame = self.dataset.eventsDF

        eventsTrainDF:DataFrame = eventsDF[0:int(len(eventsDF)/2)]
        eventsTestDF:DataFrame = eventsDF[int(len(eventsDF)/2):]

        datasetTrain = DatasetRetailRocket("rrTrain", eventsTrainDF, DataFrame(), DataFrame())

        userIDs:List[int] = list(eventsDF[Events.COL_VISITOR_ID].unique())

        recom.train(HistoryDF("test"), datasetTrain)

        counter:int = 0

        for userIdI in userIDs:
            #print("userId: " + str(userIdI))
            itemIDs:List[int] = list(
                    eventsTestDF.loc[eventsTestDF[Events.COL_VISITOR_ID] == userIdI]
                    [Events.COL_ITEM_ID].unique())
            recommendationI:List[int] = recom.recommend(userIdI, 20, args).keys()
            intersectionI:List[int] = [value for value in itemIDs if value in recommendationI]
            #print("   " + str(len(intersectionI)))
            counter += len(intersectionI)

        print("  counter: " + str(counter))


if __name__ == "__main__":
    os.chdir("..")

    dataset:ADataset = DatasetRetailRocket.readDatasetsWithFilter(minEventCount=50)

    rDscrTheMostPopular:RecommenderDescription = InputRecomRRDefinition.exportRDescTheMostPopular()
    rDscrItemBasedKNN:RecommenderDescription = InputRecomRRDefinition.exportRDescKNN()
    rDscrBPRMF:RecommenderDescription = InputRecomRRDefinition.exportRDescBPRMFIMPL()
    rDscrVMContextKNN:RecommenderDescription = InputRecomRRDefinition.exportRDescVMContextKNN()
    rDscrCosineCB:RecommenderDescription = InputRecomRRDefinition.exportRDescCosineCB()
    rDscrW2V:RecommenderDescription = InputRecomRRDefinition.exportRDescW2V()

    rDscrs:List[object] = [rDscrTheMostPopular, rDscrItemBasedKNN, rDscrBPRMF, rDscrVMContextKNN, rDscrCosineCB, rDscrW2V]
    rDscrs:List[object] = [rDscrW2V]

    for rDscrI in rDscrs:
        se:StaticEvaluation = StaticEvaluation(dataset)
        se.evaluate(rDscrI)
