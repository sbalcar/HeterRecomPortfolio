#!/usr/bin/python3

import csv
from typing import List

from datasets.rating import Rating

from pandas.core.frame import DataFrame #class

import pandas as pd

import os


class Ratings:

  COL_USERID = 'userId'
  COL_MOVIEID = 'movieId'
  COL_RATING = 'rating'
  COL_TIMESTAMP = 'timestamp'

  @staticmethod
  def readFromFileMl100k():
    ratingsFile: str = ".." + os.sep + "datasets" + os.sep + "ml-100k" + os.sep + "u.data"

    ratingsDF: DataFrame = pd.read_csv(ratingsFile, sep='\t', header=None)
    ratingsDF.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP]

    return ratingsDF


  @staticmethod
  def readFromFileMl1m():
    ratingsFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "ratings.dat"

    ratingsDF: DataFrame = pd.read_csv(ratingsFile, sep=':', usecols=[0, 2, 4, 6], header=None)
    ratingsDF.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP]

    return ratingsDF


  @staticmethod
  def readFromFileMl10M100K():
    ratingsFile: str = ".." + os.sep + "datasets" + os.sep + "ml-10M100K" + os.sep + "ratings.dat"

    ratingsDF: DataFrame = pd.read_csv(ratingsFile, sep=':', usecols=[0, 2, 4, 6], header=None)
    ratingsDF.columns = [Ratings.COL_USERID, Ratings.COL_MOVIEID, Ratings.COL_RATING, Ratings.COL_TIMESTAMP]

    return ratingsDF


  @staticmethod
  def __readFromFile(fileName:str):

      ratings:list[Rating] = []

      f = open(fileName, "r")
      for lineStrI in f:
          print(lineStrI)

          lineI = lineStrI.split()
          userIdI = int(lineI[0])
          itemIdI = int(lineI[1])
          ratingI = float(lineI[2])
          timestampI = int(lineI[3])

          ratings.append(Rating(userIdI, itemIdI, ratingI, timestampI))

      return Ratings(ratings)
