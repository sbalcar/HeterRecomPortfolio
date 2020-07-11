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

  COL_LINEAR0109 = 'linear0109'
  COL_STATIC08 = 'static08'


  @staticmethod
  def generateFileMl1m(numberOfItems:int):
    behavioursFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "behaviours.dat"

    ratingsDF:DataFrame = Ratings.readFromFileMl1m()

    uBehavStatic08Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.8])
    uBehavLinear0109Desc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1,0.9])

    behavioursDF:DataFrame = ratingsDF[[Ratings.COL_USERID, Ratings.COL_MOVIEID]].copy()

    ubStatic08s:List[List[bool]] = []
    ubLinear0109s:List[List[bool]] = []
    for indexI, rowI in behavioursDF.iterrows():
        if indexI % 1000 == 0:
            print("Generating " + str(indexI) + " / " + str(ratingsDF.shape[0]))

        ubStatic08I:List[bool] = uBehavStatic08Desc.getBehaviour(numberOfItems)
        ubLinear0109I:List[bool] = uBehavLinear0109Desc.getBehaviour(numberOfItems)

        ubStatic08s.append(ubStatic08I)
        ubLinear0109s.append(ubLinear0109I)


    behavioursDF[Behaviours.COL_STATIC08] = ubStatic08s
    behavioursDF[Behaviours.COL_LINEAR0109] = ubLinear0109s

    behavioursDF.to_csv(behavioursFile, sep='\t', index=False)


  @staticmethod
  def readFromFileMl100k():
      pass

  @staticmethod
  def readFromFileMl1m():
    behavioursFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "behaviours.dat"

    behavioursDF:DataFrame = pd.read_csv(behavioursFile, sep='\t', header=0, encoding="ISO-8859-1")
    behavioursDF.columns = [Behaviours.COL_USERID, Behaviours.COL_MOVIEID, Behaviours.COL_STATIC08, Behaviours.COL_LINEAR0109]

    behaviourStatic:List[float] = []
    behaviourLinear:List[float] = []
    for indexI, rowI in behavioursDF.iterrows():
       behavStaticI:List[str] = rowI[Behaviours.COL_STATIC08][1:-1].split(', ')
       behaviourStatic.append([(i == 'True') for i in behavStaticI])

       behavLinearI:List[str] = rowI[Behaviours.COL_LINEAR0109][1:-1].split(', ')
       behaviourLinear.append([(i == 'True') for i in behavLinearI])

    behavioursConvertedDF:DataFrame = pd.concat([behavioursDF[Behaviours.COL_USERID], behavioursDF[Behaviours.COL_MOVIEID],
                                                 Series(behaviourStatic), Series(behaviourLinear)],
                                                 axis=1, keys=[Behaviours.COL_USERID, Behaviours.COL_MOVIEID,
                                                 Behaviours.COL_STATIC08, Behaviours.COL_LINEAR0109])


    return behavioursConvertedDF

  @staticmethod
  def readFromFile10M100K():
      pass



#items = Items.readFromFile("datasets/ml-100k/u.item")
#items = Items.readFromFile("/home/stepan/workspaceJup/HeterRecomPortfolio/datasets/ml-100k/u.item")
#print(items)