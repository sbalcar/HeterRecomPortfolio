#!/usr/bin/python3

import csv
from typing import List

from datasets.ratings import Ratings

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

import pandas as pd
import numpy as np

import os

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function

class Behaviours:

  COL_USERID = Ratings.COL_USERID
  COL_MOVIEID = Ratings.COL_MOVIEID
  COL_REPETITION = 'repetition'

  COL_LINEAR0109 = 'linear0109'
  COL_STATIC08 = 'static08'


  @staticmethod
  def generateFileMl1m(numberOfItems:int, countOfRepetitions:int):
    behavioursFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "behaviours.dat"

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    uBehavStatic08Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.8])
    uBehavLinear0109Desc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1,0.9])

    ratingsCopyDF:DataFrame = ratingsDF[[Ratings.COL_USERID, Ratings.COL_MOVIEID]].copy()
    ratingsCopyDF[Behaviours.COL_REPETITION] = [range(countOfRepetitions)] * len(ratingsCopyDF)

    behavioursDF:DataFrame = ratingsCopyDF.explode(Behaviours.COL_REPETITION)
    behavioursDF[Behaviours.COL_STATIC08] = [None]*len(behavioursDF)
    behavioursDF[Behaviours.COL_LINEAR0109] = [None]*len(behavioursDF)

    behavioursDF.reset_index(inplace=True)

    numberOfRepetitionI:int
    for numberOfRepetitionI in range(countOfRepetitions):
        for indexJ, rowJ in behavioursDF.iterrows():
            if indexJ % 1000 == 0:
                print("Generating repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str(behavioursDF.shape[0]))

            repetitionIJ:int = rowJ[Behaviours.COL_REPETITION]
            if numberOfRepetitionI != repetitionIJ:
                continue

            ubStatic08IJ:List[bool] = uBehavStatic08Desc.getBehaviour(numberOfItems)
            ubLinear0109IJ:List[bool] = uBehavLinear0109Desc.getBehaviour(numberOfItems)

            behavioursDF.at[indexJ, Behaviours.COL_STATIC08] = ubStatic08IJ
            behavioursDF.at[indexJ, Behaviours.COL_LINEAR0109] = ubLinear0109IJ

    print(behavioursDF.head(10))
    del behavioursDF['index']
    print(behavioursDF.head(10))

    behavioursDF.to_csv(behavioursFile, sep='\t', index=False)


  @staticmethod
  def readFromFileMl100k():
      pass

  @staticmethod
  def readFromFileMl1m():
    behavioursFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "behaviours.dat"

    behavioursDF:DataFrame = pd.read_csv(behavioursFile, sep='\t', header=0, encoding="ISO-8859-1")
    behavioursDF.columns = [Behaviours.COL_USERID, Behaviours.COL_MOVIEID, Behaviours.COL_REPETITION,
                            Behaviours.COL_STATIC08, Behaviours.COL_LINEAR0109]

    behaviourStatic:List[float] = []
    behaviourLinear:List[float] = []
    for indexI, rowI in behavioursDF.iterrows():
       behavStaticI:List[str] = rowI[Behaviours.COL_STATIC08][1:-1].split(', ')
       behaviourStatic.append([(i == 'True') for i in behavStaticI])

       behavLinearI:List[str] = rowI[Behaviours.COL_LINEAR0109][1:-1].split(', ')
       behaviourLinear.append([(i == 'True') for i in behavLinearI])

    behavioursConvertedDF:DataFrame = pd.concat([behavioursDF[Behaviours.COL_USERID], behavioursDF[Behaviours.COL_MOVIEID],
                                                 behavioursDF[Behaviours.COL_REPETITION],
                                                 Series(behaviourStatic), Series(behaviourLinear)],
                                                 axis=1, keys=[Behaviours.COL_USERID, Behaviours.COL_MOVIEID,
                                                 Behaviours.COL_REPETITION,
                                                 Behaviours.COL_STATIC08, Behaviours.COL_LINEAR0109])

    return behavioursConvertedDF

  @staticmethod
  def readFromFile10M100K():
      pass



#items = Items.readFromFile("datasets/ml-100k/u.item")
#items = Items.readFromFile("/home/stepan/workspaceJup/HeterRecomPortfolio/datasets/ml-100k/u.item")
#print(items)