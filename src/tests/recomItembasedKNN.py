#!/usr/bin/python3

import os

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.items import Items #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class

import pandas as pd


def test01():
    print("Test 01")
    os.chdir("..")

    print("Running RecommenderItemBasedKNN:")

    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    filmsDF: DataFrame = Items.readFromFileMl1m()

    # Take only first 50k
    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:50000]

    # train recommender
    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    rec.train(pd.DataFrame(), ratingsDFTrain, pd.DataFrame(), filmsDF)

    # get one rating for update
    ratingsDFUpdate:DataFrame = ratingsDF.iloc[50005:50006]

    # get recommendations:
    print("Recommendations before update")
    r:Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})

    rec.update(ratingsDFUpdate)

    print("Recommendations after update")
    r: Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})

    print("Test for non-existent user:")
    r:Series =rec.recommend(10000, 50, {})

    print("================== END OF TEST 01 ======================\n\n\n\n\n")

def test02():
    print("Test 02")

    print("Running RecommenderItemBasedKNN:")

    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    filmsDF: DataFrame = Items.readFromFileMl1m()

    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:1000000]

    # train recommender
    rec:ARecommender = RecommenderItemBasedKNN("test", {})
    rec.train(pd.DataFrame(), ratingsDFTrain, pd.DataFrame(), filmsDF)

    r:Series = rec.recommend(1, 50, {})



if __name__ == "__main__":
    test01()
    test02()