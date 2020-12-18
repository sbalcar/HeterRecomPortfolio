#!/usr/bin/python3

import os

from recommender.aRecommender import ARecommender #class
from recommender.recommenderTheMostSold import RecommenderTheMostSold #class

from datasets.retailrocket.events import Events #class

from pandas.core.frame import DataFrame #class


if __name__ == "__main__":

    os.chdir("..")

    eventsDF:DataFrame = Events.readFromFile()

    r:ARecommender = RecommenderTheMostSold("rTheMostSold", {})
    r.train(None, eventsDF, None, None)

    recommendation = r.recommend(1, 20, {})
    print(recommendation)