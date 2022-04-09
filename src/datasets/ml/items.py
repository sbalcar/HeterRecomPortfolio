#!/usr/bin/python3

import io
import csv
from typing import List

from pandas.core.frame import DataFrame #class

import pandas as pd

import os

class Items:
  # 100 KB
  # 1 | Toy Story(1995) | 01 - Jan - 1995 | | http: // us.imdb.com / M / title - exact?Toy % 20 Story % 20(1995) | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
  # movie id | movie title | release date | video releas date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |'''

  COL_MOVIEID:str = 'movieId'
  COL_MOVIETITLE:str = 'movieTitle'
  COL_RELEASEDATE:str = 'releaseDate'
  COL_VIDEORELEASEDATE:str = 'videoReleaseDate'
  COL_IMDbURL:str = 'IMDbURL'

  # 1 m
  # MovieID::Title::Genres

  COL_GENRES:str = 'Genres'

  GENRE_ACTION:str = "Action"
  GENRE_ADVENTURE:str = "Adventure"
  GENRE_ANIMATION:str = "Animation"
  GENRE_CHILDRENS:str = "Children's"
  GENRE_COMEDY:str = "Comedy"
  GENRE_CRIME:str = "Crime"
  GENRE_DOCUMENTARY:str = "Documentary"
  GENRE_DRAMA:str = "Drama"
  GENRE_FANTASY:str = "Fantasy"
  GENRE_FILM_NOIR:str = "Film-Noir"
  GENRE_HORROR:str = "Horror"
  GENRE_MUSICAL:str = "Musical"
  GENRE_MYSTERY:str = "Mystery"
  GENRE_ROMANCE:str = "Romance"
  GENRE_SCIFI:str = "Sci-Fi"
  GENRE_THRILLER:str = "Thriller"
  GENRE_WAR:str = "War"
  GENRE_WESTERN:str = "Western"


  @staticmethod
  def getAllGenres():
    return [
      Items.GENRE_ACTION,
      Items.GENRE_ADVENTURE,
      Items.GENRE_ANIMATION,
      Items.GENRE_CHILDRENS,
      Items.GENRE_COMEDY,
      Items.GENRE_CRIME,
      Items.GENRE_DOCUMENTARY,
      Items.GENRE_DRAMA,
      Items.GENRE_FANTASY,
      Items.GENRE_FILM_NOIR,
      Items.GENRE_HORROR,
      Items.GENRE_MUSICAL,
      Items.GENRE_MYSTERY,
      Items.GENRE_ROMANCE,
      Items.GENRE_SCIFI,
      Items.GENRE_THRILLER,
      Items.GENRE_WAR,
      Items.GENRE_WESTERN]

  @staticmethod
  def countA(ratingsDF:DataFrame, itemIds:List[int]):

    genreCountDict:dict = {gI: 0 for gI in Items.getAllGenres()}

    itemIdI:int
    for itemIdI in itemIds:
      rowI = ratingsDF[ratingsDF[Items.COL_MOVIEID] == itemIdI]
      genresStrI:str = str(rowI.iloc[0][Items.COL_GENRES])
      #print(genresStrI)

      genresI:List[str] = genresStrI.split('|')
      for genreJ in genresI:
        genreCountDict[genreJ] += 1.0/len(genresI)

    return genreCountDict

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

    itemsDF: DataFrame = pd.read_csv(itemsFile, sep='::', usecols=[0, 1, 2], header=None, encoding="ISO-8859-1", engine='python')
    itemsDF.columns = [Items.COL_MOVIEID, Items.COL_MOVIETITLE, Items.COL_GENRES]

    return itemsDF


#  @staticmethod
#  def readFromFileMl1m():
#      itemsFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "movies.dat"
#
#      with open(itemsFile, encoding='latin-1') as f:
#        str = " ".join([l.rstrip() + '\n' for l in f])
#
#      str = str.replace("::", "@")
#
#      itemsDF: DataFrame = pd.read_table(io.StringIO(str), sep='@', usecols=[0, 1, 2], header=None, engine='python', encoding='latin-1')
#      itemsDF.columns = [Items.COL_MOVIEID, Items.COL_MOVIETITLE, Items.COL_GENRES]
#
#      return itemsDF


  @staticmethod
  def readFromFile10M100K():
      pass



#items = Items.readFromFile("datasets/ml-100k/u.item")
#items = Items.readFromFile("/home/stepan/workspaceJup/HeterRecomPortfolio/datasets/ml-100k/u.item")
#print(items)