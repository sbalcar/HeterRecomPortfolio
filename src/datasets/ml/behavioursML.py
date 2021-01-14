#!/usr/bin/python3

from typing import List

from datasets.ml.ratings import Ratings

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

import pandas as pd
import numpy as np

import os
import random

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function



class BehavioursML:

  COL_USERID = Ratings.COL_USERID
  COL_MOVIEID = Ratings.COL_MOVIEID
  COL_REPETITION = 'repetition'
  COL_BEHAVIOUR = 'behaviour'

  BHVR_LINEAR0109 = 'linear0109'
  BHVR_STATIC08 = 'static08'
  BHVR_STATIC06 = 'static06'
  BHVR_STATIC04 = 'static04'
  BHVR_STATIC02 = 'static02'


  @staticmethod
  def getColNameUserID():
    return BehavioursML.COL_USERID

  @staticmethod
  def getColNameItemID():
    return BehavioursML.COL_MOVIEID

  @staticmethod
  def getColNameRepetition():
    return BehavioursML.COL_REPETITION

  @staticmethod
  def getColNameBehaviour():
    return BehavioursML.COL_BEHAVIOUR


  @staticmethod
  def getFile(behaviourID:str):
      return ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "behaviour" + behaviourID + ".dat"


  @staticmethod
  def generateFileMl1m(numberOfItems:int, countOfRepetitions:int, behaviourID:str, uBehavDesc:UserBehaviourDescription):

      np.random.seed(42)
      random.seed(42)

      print("Generate Behaviour ML " + behaviourID)

      behaviourFile:str = BehavioursML.getFile(behaviourID)

      ratingsDF:DataFrame = Ratings.readFromFileMl1m()

      ratingsCopyDF:DataFrame = ratingsDF[[Ratings.COL_USERID, Ratings.COL_MOVIEID]].copy()
      ratingsCopyDF[BehavioursML.COL_REPETITION] = [range(countOfRepetitions)] * len(ratingsCopyDF)

      behavioursDF:DataFrame = ratingsCopyDF.explode(BehavioursML.COL_REPETITION)
      behavioursDF[BehavioursML.COL_BEHAVIOUR] = [None] * len(behavioursDF)
      behavioursDF.reset_index(inplace=True)

      if behaviourID is BehavioursML.BHVR_LINEAR0109:
          #print(Behaviours.BHVR_LINEAR0109)
          BehavioursML.__generateLinear0109BehaviourMl1m(behavioursDF, numberOfItems, countOfRepetitions, uBehavDesc)
      elif behaviourID is BehavioursML.BHVR_STATIC08:
          #print(Behaviours.BHVR_STATIC08)
          BehavioursML.__generateStatic08BehaviourMl1m(behavioursDF, numberOfItems, countOfRepetitions, uBehavDesc)
      else:
          #print("General")
          BehavioursML.__generateGeneralBehaviourMl1m(behavioursDF, numberOfItems, countOfRepetitions, uBehavDesc)

      print(behavioursDF.head(10))
      del behavioursDF['index']
      print(behavioursDF.head(10))

      behavioursDF.to_csv(behaviourFile, sep='\t', index=False)


  @staticmethod
  def __generateLinear0109BehaviourMl1m(behavioursDF:DataFrame, numberOfItems:int, countOfRepetitions:int,
                                        uBehavLinear0109Desc:UserBehaviourDescription):

      uBehavStatic08Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.8])

      numberOfRepetitionI:int
      for numberOfRepetitionI in range(countOfRepetitions):
          for indexJ, rowJ in behavioursDF.iterrows():
              if indexJ % 1000 == 0:
                  print("Generating repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str(behavioursDF.shape[0]))

              repetitionIJ:int = rowJ[BehavioursML.COL_REPETITION]
              if numberOfRepetitionI != repetitionIJ:
                  continue

              # generation "ubStatic08IJ" ensures backward compatibility of results
              ubStatic08IJ:List[bool] = uBehavStatic08Desc.getBehaviour(numberOfItems)
              if ubStatic08IJ is None: print("aaa")
              ubLinear0109IJ:List[bool] = uBehavLinear0109Desc.getBehaviour(numberOfItems)

              strLinear0109IJ:str =  BehavioursML.__convertToString(ubLinear0109IJ)
              #print(strLinear0109IJ)
              behavioursDF.at[indexJ, BehavioursML.COL_BEHAVIOUR] = strLinear0109IJ


  @staticmethod
  def __generateStatic08BehaviourMl1m(behavioursDF:DataFrame, numberOfItems:int, countOfRepetitions:int,
                                        uBehavStatic08Desc:UserBehaviourDescription):

      uBehavLinear0109Desc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1, 0.9])

      numberOfRepetitionI:int
      for numberOfRepetitionI in range(countOfRepetitions):
          for indexJ, rowJ in behavioursDF.iterrows():
              if indexJ % 1000 == 0:
                  print("Generating repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str(behavioursDF.shape[0]))

              repetitionIJ:int = rowJ[BehavioursML.COL_REPETITION]
              if numberOfRepetitionI != repetitionIJ:
                  continue

              ubStatic08IJ:List[bool] = uBehavStatic08Desc.getBehaviour(numberOfItems)
              # generation "ubLinear0109IJ" ensures backward compatibility of results
              ubLinear0109IJ:List[bool] = uBehavLinear0109Desc.getBehaviour(numberOfItems)
              if ubLinear0109IJ is None: print("aaa")

              strStatic08IJ:str = BehavioursML.__convertToString(ubStatic08IJ)
              #print(strStatic08IJ)

              behavioursDF.at[indexJ, BehavioursML.COL_BEHAVIOUR] = strStatic08IJ


  @staticmethod
  def __generateGeneralBehaviourMl1m(behavioursDF:DataFrame, numberOfItems:int, countOfRepetitions:int,
                                     uBehavDesc:UserBehaviourDescription):

      for numberOfRepetitionI in range(countOfRepetitions):
          for indexJ, rowJ in behavioursDF.iterrows():
              if indexJ % 1000 == 0:
                  print("Generating repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str(behavioursDF.shape[0]))

              repetitionIJ:int = rowJ[BehavioursML.COL_REPETITION]
              if numberOfRepetitionI != repetitionIJ:
                  continue

              uBehavIJ:List[bool] = uBehavDesc.getBehaviour(numberOfItems)
              strBehavIJ:str = BehavioursML.__convertToString(uBehavIJ)
              behavioursDF.at[indexJ, BehavioursML.COL_BEHAVIOUR] = strBehavIJ


  @staticmethod
  def convertToString(values:List[bool]):
      intValues:List[int] = list(map(int, values))
      strValues:List[str] = list(map(str, intValues))

      strOfBooleans:str = str("".join(strValues))
      return "b" + strOfBooleans

  @staticmethod
  def convertToListOfBoolean(string:str):
      stringValues:List[str] = []
      stringValues[:0] = string[1:]
      #print(stringValues)
      intValues:List[int] = list(map(int, stringValues))

      boolValues:List[bool] = [xI == 1 for xI in intValues]
      return boolValues


  @staticmethod
  def readFromFileMl100k():
      pass

  @staticmethod
  def readFromFileMl1m(behavioursFile:str):

    behavioursDF:DataFrame = pd.read_csv(behavioursFile, sep='\t', header=0, encoding="ISO-8859-1")
    behavioursDF.columns = [BehavioursML.COL_USERID, BehavioursML.COL_MOVIEID, BehavioursML.COL_REPETITION,
                            BehavioursML.COL_BEHAVIOUR]

    return behavioursDF

#    behaviour:List[float] = []
#    for indexI, rowI in behavioursDF.iterrows():
#
#       behaviourI:List[bool] = Behaviours.__convertToListOfBoolean(str(rowI[Behaviours.COL_BEHAVIOUR]))
#       behaviour.append(behaviourI)
#
#    behavioursConvertedDF:DataFrame = pd.concat([behavioursDF[Behaviours.COL_USERID], behavioursDF[Behaviours.COL_MOVIEID],
#                                                 behavioursDF[Behaviours.COL_REPETITION], Series(behaviour)],
#                                                 axis=1, keys=[Behaviours.COL_USERID, Behaviours.COL_MOVIEID,
#                                                 Behaviours.COL_REPETITION, Behaviours.COL_BEHAVIOUR])
#
#    return behavioursConvertedDF

  @staticmethod
  def readFromFile10M100K():
      pass



#items = Items.readFromFile("datasets/ml-100k/u.item")
#items = Items.readFromFile("/home/stepan/workspaceJup/HeterRecomPortfolio/datasets/ml-100k/u.item")
#print(items)