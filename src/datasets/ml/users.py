#!/usr/bin/python3

import csv
import io
from typing import List

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
      usersDF.columns = [Users.COL_USERID, Users.COL_GENDER, Users.COL_AGE, Users.COL_OCCUPATION, Users.COL_ZIPCODE]

      return usersDF


  @staticmethod
  def readFromFileMl1m():
      usersFile: str = ".." + os.sep + "datasets" + os.sep + "ml-1m" + os.sep + "users.dat"
      #usersFile: str = ".." + os.sep + "datasets" + os.sep + "ml-10M100K" + os.sep + "tags.dat"

      usersDF:DataFrame = pd.read_csv(usersFile, sep=':', usecols=[0, 2, 4, 6, 8], header=None)
      usersDF.columns = [Users.COL_USERID, Users.COL_GENDER, Users.COL_AGE, Users.COL_OCCUPATION, Users.COL_ZIPCODE]

      return usersDF



  @staticmethod
  def readFromFile10M100K():
      pass

