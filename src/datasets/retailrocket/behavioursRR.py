#!/usr/bin/python3

from typing import List

from datasets.ml.ratings import Ratings

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

from datasets.ml.behavioursML import BehavioursML

from datasets.retailrocket.events import Events #class

import pandas as pd
import numpy as np

import os
import random

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function



class BehavioursRR:

  COL_USERID:str = Events.COL_VISITOR_ID
  COL_ITEMID:str = Events.COL_ITEM_ID
  COL_REPETITION:str = BehavioursML.COL_REPETITION
  COL_BEHAVIOUR:str = BehavioursML.COL_BEHAVIOUR

  BHVR_LINEAR0109:str = BehavioursML.BHVR_LINEAR0109
  BHVR_STATIC08:str = BehavioursML.BHVR_STATIC08
  BHVR_STATIC06:str = BehavioursML.BHVR_STATIC06
  BHVR_STATIC04:str = BehavioursML.BHVR_STATIC04
  BHVR_STATIC02:str = BehavioursML.BHVR_STATIC02

  BHVR_POWERLAW054MIN048:str = BehavioursML.BHVR_POWERLAW054MIN048

  @staticmethod
  def getColNameUserID():
    return BehavioursRR.COL_USERID

  @staticmethod
  def getColNameItemID():
    return BehavioursRR.COL_ITEMID

  @staticmethod
  def getColNameRepetition():
    return BehavioursRR.COL_REPETITION

  @staticmethod
  def getColNameBehaviour():
    return BehavioursRR.COL_BEHAVIOUR


  @staticmethod
  def getFile(behaviourID:str):
      return ".." + os.sep + "datasets" + os.sep + "retailrocket" + os.sep + "behaviour" + behaviourID + ".dat"


  @staticmethod
  def generateFileRR(numberOfItems:int, countOfRepetitions:int, behaviourID:str, uBehavDesc:UserBehaviourDescription):

      np.random.seed(42)
      random.seed(42)

      print("Generate Behaviour RR " + behaviourID)

      behaviourFile:str = BehavioursRR.getFile(behaviourID)

      eventsDF:DataFrame = Events.readFromFile()

      eventsCopyDF:DataFrame = eventsDF[[Events.COL_VISITOR_ID, Events.COL_ITEM_ID]].copy()
      eventsCopyDF[BehavioursRR.COL_REPETITION] = [range(countOfRepetitions)] * len(eventsCopyDF)

      behavioursDF:DataFrame = eventsCopyDF.explode(BehavioursRR.COL_REPETITION)
      behavioursDF[BehavioursRR.COL_BEHAVIOUR] = [None]*len(behavioursDF)
      behavioursDF.reset_index(inplace=True)

      #print("General")
      BehavioursRR.__generateGeneralBehaviourrRR(behavioursDF, numberOfItems, countOfRepetitions, uBehavDesc)

      #print(behavioursDF.head(10))
      del behavioursDF['index']
      #print(behavioursDF.head(10))

      behavioursDF.to_csv(behaviourFile, sep='\t', index=False)


  @staticmethod
  def __generateGeneralBehaviourrRR(behavioursDF :DataFrame, numberOfItems :int, countOfRepetitions :int,
                                   uBehavDesc :UserBehaviourDescription):

    for numberOfRepetitionI in range(countOfRepetitions):
        for indexJ, rowJ in behavioursDF.iterrows():
            if indexJ % 1000 == 0:
                print("Generating repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str
                    (behavioursDF.shape[0]))

            repetitionIJ :int = rowJ[BehavioursRR.COL_REPETITION]
            if numberOfRepetitionI != repetitionIJ:
                continue

            uBehavIJ :List[bool] = uBehavDesc.getBehaviour(numberOfItems)
            strBehavIJ :str = BehavioursML.convertToString(uBehavIJ)
            behavioursDF.at[indexJ, BehavioursRR.COL_BEHAVIOUR] = strBehavIJ


  @staticmethod
  def readFromFileRR(behavioursFile:str):

    behavioursDF:DataFrame = pd.read_csv(behavioursFile, sep='\t', header=0, encoding="ISO-8859-1")
    behavioursDF.columns = [BehavioursRR.COL_USERID, BehavioursRR.COL_ITEMID, BehavioursRR.COL_REPETITION,
                            BehavioursRR.COL_BEHAVIOUR]

 #   for i, bI in behavioursDF.iterrows():
 #       #print(bI[BehavioursRR.COL_BEHAVIOUR])
 #       a = BehavioursML.convertToListOfBoolean(bI[BehavioursRR.COL_BEHAVIOUR])
 #       print(a)
 #       behavioursDF.loc[i, BehavioursRR.COL_BEHAVIOUR] = a


    return behavioursDF
