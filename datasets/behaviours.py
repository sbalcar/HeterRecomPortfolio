#!/usr/bin/python3

import csv
from typing import List

from datasets.ratings import Ratings

from pandas.core.frame import DataFrame #class
from pandas.core.frame import Series #class

import pandas as pd
import numpy as np

import os
import random

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function



class Behaviours:

  COL_USERID = Ratings.COL_USERID
  COL_MOVIEID = Ratings.COL_MOVIEID
  COL_REPETITION = 'repetition'

  COL_LINEAR0109 = 'linear0109'
  COL_STATIC08 = 'static08'
  COL_STATIC06 = 'static06'
  COL_STATIC04 = 'static04'
  COL_STATIC02 = 'static02'

  @staticmethod
  def generateFileMl1m(numberOfItems:int, countOfRepetitions:int):

      np.random.seed(42)
      random.seed(42)

      behavioursFile:str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "behaviours.dat"

      ratingsDF:DataFrame = Ratings.readFromFileMl1m()

      uBehavStatic08Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.8])

      uBehavLinear0109Desc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1,0.9])

      ratingsCopyDF:DataFrame = ratingsDF[[Ratings.COL_USERID, Ratings.COL_MOVIEID]].copy()
      ratingsCopyDF[Behaviours.COL_REPETITION] = [range(countOfRepetitions)] * len(ratingsCopyDF)

      behavioursDF:DataFrame = ratingsCopyDF.explode(Behaviours.COL_REPETITION)
      behavioursDF[Behaviours.COL_STATIC08] = [None]*len(behavioursDF)
      behavioursDF[Behaviours.COL_STATIC06] = [None]*len(behavioursDF)
      behavioursDF[Behaviours.COL_STATIC04] = [None]*len(behavioursDF)
      behavioursDF[Behaviours.COL_STATIC02] = [None]*len(behavioursDF)
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

              strStatic08IJ:str = Behaviours.__convertToString(ubStatic08IJ)
              strLinear0109IJ:str =  Behaviours.__convertToString(ubLinear0109IJ)

              behavioursDF.at[indexJ, Behaviours.COL_STATIC08] = strStatic08IJ
              behavioursDF.at[indexJ, Behaviours.COL_LINEAR0109] = strLinear0109IJ


      
      np.random.seed(42)
      random.seed(42)

      uBehavStatic06Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.6])

      for numberOfRepetitionI in range(countOfRepetitions):
          for indexJ, rowJ in behavioursDF.iterrows():
              if indexJ % 1000 == 0:
                  print("Generating " + Behaviours.COL_STATIC06 + " repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str(behavioursDF.shape[0]))

              repetitionIJ:int = rowJ[Behaviours.COL_REPETITION]
              if numberOfRepetitionI != repetitionIJ:
                  continue

              ubStatic06IJ:List[bool] = uBehavStatic06Desc.getBehaviour(numberOfItems)
              strStatic06IJ:str = Behaviours.__convertToString(ubStatic06IJ)
              behavioursDF.at[indexJ, Behaviours.COL_STATIC06] = strStatic06IJ



      np.random.seed(42)
      random.seed(42)

      uBehavStatic04Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.4])

      for numberOfRepetitionI in range(countOfRepetitions):
          for indexJ, rowJ in behavioursDF.iterrows():
              if indexJ % 1000 == 0:
                  print("Generating " + Behaviours.COL_STATIC04 + " repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str(behavioursDF.shape[0]))

              repetitionIJ:int = rowJ[Behaviours.COL_REPETITION]
              if numberOfRepetitionI != repetitionIJ:
                  continue

              ubStatic04IJ:List[bool] = uBehavStatic04Desc.getBehaviour(numberOfItems)
              strStatic04IJ:str = Behaviours.__convertToString(ubStatic04IJ)
              behavioursDF.at[indexJ, Behaviours.COL_STATIC04] = strStatic04IJ



      np.random.seed(42)
      random.seed(42)

      uBehavStatic02Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.2])

      for numberOfRepetitionI in range(countOfRepetitions):
          for indexJ, rowJ in behavioursDF.iterrows():
              if indexJ % 1000 == 0:
                  print("Generating " + Behaviours.COL_STATIC02 + " repetition " + str(numberOfRepetitionI) + "   " + str(indexJ) + " / " + str(behavioursDF.shape[0]))

              repetitionIJ:int = rowJ[Behaviours.COL_REPETITION]
              if numberOfRepetitionI != repetitionIJ:
                  continue

              ubStatic02IJ:List[bool] = uBehavStatic02Desc.getBehaviour(numberOfItems)
              strStatic02IJ:str = Behaviours.__convertToString(ubStatic02IJ)
              behavioursDF.at[indexJ, Behaviours.COL_STATIC02] = strStatic02IJ
      


      print(behavioursDF.head(10))
      del behavioursDF['index']
      print(behavioursDF.head(10))

      #df = df.astype({'col_name_2': 'float64', 'col_name_3': 'float64'})
      behavioursDF.to_csv(behavioursFile, sep='\t', index=False)


  @staticmethod
  def __convertToString(values:List[bool]):
      intValues:List[int] = list(map(int, values))
      strValues:List[str] = list(map(str, intValues))

      strOfBooleans:str = str("".join(strValues))
      return "b" + strOfBooleans

  @staticmethod
  def __convertToListOfBoolean(string:str):
      stringValues:List[str] = []
      stringValues[:0] = string[1:]
      intValues:List[int] = list(map(int, stringValues))

      boolValues:List[bool] = [xI == 1 for xI in intValues]
      return boolValues


  @staticmethod
  def readFromFileMl100k():
      pass

  @staticmethod
  def readFromFileMl1m():
    behavioursFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "behaviours.dat"

    behavioursDF:DataFrame = pd.read_csv(behavioursFile, sep='\t', header=0, encoding="ISO-8859-1")
    behavioursDF.columns = [Behaviours.COL_USERID, Behaviours.COL_MOVIEID, Behaviours.COL_REPETITION,
                            Behaviours.COL_STATIC08, Behaviours.COL_STATIC06, Behaviours.COL_STATIC04,
                            Behaviours.COL_STATIC02, Behaviours.COL_LINEAR0109]


    behaviourStatic08:List[float] = []
    behaviourStatic06:List[float] = []
    behaviourStatic04:List[float] = []
    behaviourStatic02:List[float] = []
    behaviourLinear0109:List[float] = []
    for indexI, rowI in behavioursDF.iterrows():

       behavStatic08I:List[bool] = Behaviours.__convertToListOfBoolean(str(rowI[Behaviours.COL_STATIC08]))
       behaviourStatic08.append(behavStatic08I)
       
       behavStatic06I:List[bool] = Behaviours.__convertToListOfBoolean(str(rowI[Behaviours.COL_STATIC06]))
       behaviourStatic06.append(behavStatic06I)

       behavStatic04I:List[bool] = Behaviours.__convertToListOfBoolean(str(rowI[Behaviours.COL_STATIC04]))
       behaviourStatic04.append(behavStatic04I)

       behavStatic02I:List[bool] = Behaviours.__convertToListOfBoolean(str(rowI[Behaviours.COL_STATIC02]))
       behaviourStatic02.append(behavStatic02I)
       
       behavLinear0109I:List[bool] = Behaviours.__convertToListOfBoolean(str(rowI[Behaviours.COL_LINEAR0109]))
       behaviourLinear0109.append(behavLinear0109I)

    behavioursConvertedDF:DataFrame = pd.concat([behavioursDF[Behaviours.COL_USERID], behavioursDF[Behaviours.COL_MOVIEID],
                                                 behavioursDF[Behaviours.COL_REPETITION],
                                                 Series(behaviourStatic08), Series(behaviourStatic06),
                                                 Series(behaviourStatic04), Series(behaviourStatic02),
                                                 Series(behaviourLinear0109)],
                                                 axis=1, keys=[Behaviours.COL_USERID, Behaviours.COL_MOVIEID,
                                                 Behaviours.COL_REPETITION,
                                                 Behaviours.COL_STATIC08, Behaviours.COL_STATIC06, Behaviours.COL_STATIC04,
                                                 Behaviours.COL_STATIC02, Behaviours.COL_LINEAR0109])

    return behavioursConvertedDF

  @staticmethod
  def readFromFile10M100K():
      pass



#items = Items.readFromFile("datasets/ml-100k/u.item")
#items = Items.readFromFile("/home/stepan/workspaceJup/HeterRecomPortfolio/datasets/ml-100k/u.item")
#print(items)