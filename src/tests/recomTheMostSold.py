#!/usr/bin/python3

import os

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostSold import RecommenderTheMostSold #class

from datasets.aDataset import ADataset #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.retailrocket.events import Events #class

from history.historyDF import HistoryDF #class

from pandas.core.frame import DataFrame #class


if __name__ == "__main__":

    os.chdir("..")

    eventsDF:DataFrame = Events.readFromFile()

    dataset:ADataset = DatasetRetailRocket(eventsDF, DataFrame(), DataFrame())

    r:ARecommender = RecommenderTheMostSold("rTheMostSold", {})
    r.train(HistoryDF("test"), dataset)

    recommendation = r.recommend(1, 20, {})
    print(recommendation)