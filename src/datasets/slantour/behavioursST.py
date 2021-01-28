#!/usr/bin/python3

from typing import List

from datasets.ml.ratings import Ratings

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

from datasets.ml.behavioursML import BehavioursML

from datasets.slantour.events import Events #class

import pandas as pd
import numpy as np

import os
import random

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function



class BehavioursST:

  COL_VISIT_ID = Events.COL_VISIT_ID
  COL_REPETITION = BehavioursML.COL_REPETITION
  COL_BEHAVIOUR = BehavioursML.COL_BEHAVIOUR

  BHVR_LINEAR0109 = BehavioursML.BHVR_LINEAR0109
  BHVR_STATIC08 = BehavioursML.BHVR_STATIC08
  BHVR_STATIC06 = BehavioursML.BHVR_STATIC06
  BHVR_STATIC04 = BehavioursML.BHVR_STATIC04
  BHVR_STATIC02 = BehavioursML.BHVR_STATIC02

  BHVR_POWERLAW054MIN048 = BehavioursML.BHVR_POWERLAW054MIN048

#  @staticmethod
#  def getColNameUserID():
#    return BehavioursST.COL_USERID

#  @staticmethod
#  def getColNameItemID():
#    return BehavioursST.COL_ITEMID

  @staticmethod
  def getColNameRepetition():
    return BehavioursST.COL_REPETITION

  @staticmethod
  def getColNameBehaviour():
    return BehavioursST.COL_BEHAVIOUR


  @staticmethod
  def getFile(behaviourID:str):
      return ".." + os.sep + "datasets" + os.sep + "slantour" + os.sep + "behaviour" + behaviourID + ".dat"


  @staticmethod
  def generateFileST(numberOfItems:int, countOfRepetitions:int, behaviourID:str, uBehavDesc:UserBehaviourDescription):

      np.random.seed(42)
      random.seed(42)

      print("Generate Behaviour ST " + behaviourID)

      behaviourFile:str = BehavioursST.getFile(behaviourID)

      eventsDF:DataFrame = Events.readFromFile()

      eventsCopyDF:DataFrame = eventsDF[[Events.COL_VISIT_ID]].copy()
      eventsCopyDF[BehavioursST.COL_REPETITION] = [range(countOfRepetitions)] * len(eventsCopyDF)

      behavioursDF:DataFrame = eventsCopyDF.explode(BehavioursST.COL_REPETITION)
      behavioursDF[BehavioursST.COL_BEHAVIOUR] = [None]*len(behavioursDF)
      behavioursDF.reset_index(inplace=True)

      #print("General")
      BehavioursST.__generateGeneralBehaviourrST(behavioursDF, numberOfItems, countOfRepetitions, uBehavDesc)

      print(behavioursDF.head(10))
      del behavioursDF['index']
      print(behavioursDF.head(10))

      behavioursDF.to_csv(behaviourFile, sep='\t', index=False)


  @staticmethod
  def __generateGeneralBehaviourrST(behavioursDF :DataFrame, numberOfItems :int, countOfRepetitions :int,
                                   uBehavDesc :UserBehaviourDescription):

    for numberOfRepetitionI in range(countOfRepetitions):
        for indexJ, rowJ in behavioursDF.iterrows():
            if indexJ % 1000 == 0:
                print("Generating repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str
                    (behavioursDF.shape[0]))

            repetitionIJ :int = rowJ[BehavioursST.COL_REPETITION]
            if numberOfRepetitionI != repetitionIJ:
                continue

            uBehavIJ :List[bool] = uBehavDesc.getBehaviour(numberOfItems)
            strBehavIJ :str = BehavioursML.convertToString(uBehavIJ)
            behavioursDF.at[indexJ, BehavioursST.COL_BEHAVIOUR] = strBehavIJ


  @staticmethod
  def readFromFileST(behavioursFile:str):

    behavioursDF:DataFrame = pd.read_csv(behavioursFile, sep='\t', header=0, encoding="ISO-8859-1")
#    behavioursDF.columns = [BehavioursST.COL_VISIT_ID, BehavioursST.COL_REPETITION, BehavioursST.COL_BEHAVIOUR]

    return behavioursDF

