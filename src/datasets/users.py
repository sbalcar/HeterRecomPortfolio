#!/usr/bin/python3

import csv
from typing import List

from datasets.user import User

from pandas.core.frame import DataFrame #class

import pandas as pd

import os

class Users:

  COL_USERID = 'userId'
  COL_AGE = 'age'
  COL_GENDER = 'gender'
  COL_OCCUPATION = 'occupation'
  COL_ZIPCODE = 'zipCode'

  @staticmethod
  def readFromFileMl100k():
      usersFile: str = ".." + os.sep + "datasets" + os.sep + "ml-100k" + os.sep + "u.user"

      usersDF:DataFrame = pd.read_csv(usersFile, sep='|', header=None)
      usersDF.columns = [Users.COL_USERID, Users.COL_AGE, Users.COL_GENDER, Users.COL_OCCUPATION, Users.COL_ZIPCODE]

      return usersDF


  @staticmethod
  def readFromFileMl1m():
      usersFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "users.dat"
      #usersFile: str = ".." + os.sep + "datasets" + os.sep + "ml-10M100K" + os.sep + "tags.dat"

      usersDF:DataFrame = pd.read_csv(usersFile, sep=':', usecols=[0, 2, 4, 6, 8], header=None)
      usersDF.columns = [Users.COL_USERID, Users.COL_AGE, Users.COL_GENDER, Users.COL_OCCUPATION, Users.COL_ZIPCODE]

      return usersDF


  @staticmethod
  def readFromFile10M100K():
      pass


  @staticmethod
  def __readFromFile(fileName:str):

      users:list[Users] = []

      f = open(fileName, "r")
      for lineStrI in f:
          print(lineStrI)

          lineI = lineStrI.split('|')

          userIdI:int = int(lineI[0])
          ageI:int = int(lineI[1])
          sexI:str = str(lineI[2])
          occupationI:str = str(lineI[3])
          zipCodeI:str = str(lineI[4].strip())

          users.append(User(userIdI, ageI, sexI, occupationI, zipCodeI))

      return Users(users)


