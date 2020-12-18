#!/usr/bin/python3
import sys 
  
# appending a path 
sys.path.append('..')
 
import os

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class
from pandas.core.series import Series #class

from datasets.ml.ratings import Ratings #class
from datasets.ml.items import Items #class
from datasets.ml.users import Users #class

from recommender.aRecommender import ARecommender #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

import pandas as pd


def test01():
    print("Test 01")
    os.chdir("..")

    print("Running Recommender BPRMF:")

    ratingsDF: DataFrame = Ratings.readFromFileMl1m()

    itemsDF: DataFrame = Items.readFromFileMl1m()
    usersDF: DataFrame = Users.readFromFileMl1m()

    # Take only first 500k
    ratingsDFTrain:DataFrame = ratingsDF.iloc[0:500000]

    # train recommender
    rec:ARecommender = RecommenderBPRMF("test", {})
    rec.train(pd.DataFrame(), ratingsDFTrain, usersDF, itemsDF)

    # get one rating for update
    ratingsDFUpdate:DataFrame = ratingsDF.iloc[500005:504006]

    # get recommendations:
    print("Recommendations before update")
    r:Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})
    print(r);
    for i in  range(ratingsDFUpdate.shape[0]):
        rUp = ratingsDFUpdate.iloc[i:i+1,:]
        rec.update(rUp)

    print("Recommendations after update")
    r: Series = rec.recommend(ratingsDFUpdate['userId'].iloc[0], 50, {})
    print(r);
    
    print("Test for non-existent user:")
    r:Series =rec.recommend(10000, 50, {})
    print(r);
    
    print("================== END OF TEST 01 ======================\n\n\n\n\n")


if __name__ == "__main__":
    test01()
