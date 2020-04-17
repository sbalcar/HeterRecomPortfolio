#!/usr/bin/python3

import csv
from typing import List

from datasets.item import Item

from pandas.core.frame import DataFrame #class

import pandas as pd

import os

class Items:
  # 100 KB
  # 1 | Toy Story(1995) | 01 - Jan - 1995 | | http: // us.imdb.com / M / title - exact?Toy % 20 Story % 20(1995) | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
  # movie id | movie title | release date | video releas date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |'''

  COL_MOVIEID = 'movieId'
  COL_MOVIETITLE = 'movieTitle'
  COL_RELEASEDATE = 'releaseDate'
  COL_VIDEORELEASEDATE = 'videoReleaseDate'
  COL_IMDbURL = 'IMDbURL'

  # 1 m
  # MovieID::Title::Genres

  COL_GENRES = 'Genres'


  @staticmethod
  def readFromFileMl100k():
    itemsFile: str = ".." + os.sep + "datasets" + os.sep + "ml-100k" + os.sep + "u.item"

    itemsDF: DataFrame = pd.read_csv(itemsFile, sep='\t', header=None, encoding="ISO-8859-1")
#    itemsDF.columns = [Items.COL_MOVIEID, Items.COL_MOVIETITLE, Items.COL_RELEASEDATE, Items.COL_VIDEORELEASEDATE,
#                       Items.COL_IMDbURL, ]

    return itemsDF

  @staticmethod
  def readFromFileMl1m():
    itemsFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "movies.dat"

    itemsDF: DataFrame = pd.read_csv(itemsFile, sep=':', usecols=[0, 2, 4], header=None, encoding="ISO-8859-1")
    itemsDF.columns = [Items.COL_MOVIEID, Items.COL_MOVIETITLE, Items.COL_GENRES]

    return itemsDF

  @staticmethod
  def readFromFile10M100K():
      pass


  @staticmethod
  def __readFromFile(fileName:str):

      items:list[Item] = []

      f = open(fileName, "r", encoding='ISO-8859-1')
      for lineStrI in f:
         lineI = lineStrI.split('|')
         itemIdI = int(lineI[0])
         nameI = lineI[1]

         items.append(Item(itemIdI, nameI))

      return items



#items = Items.readFromFile("datasets/ml-100k/u.item")
#items = Items.readFromFile("/home/stepan/workspaceJup/HeterRecomPortfolio/datasets/ml-100k/u.item")
#print(items)