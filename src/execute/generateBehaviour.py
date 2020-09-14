#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class


def generateBehaviour():
   print("Generate Behaviour")

   np.random.seed(42)
   random.seed(42)

   countOfItems:int = 20
   countOfRepetitions:int = 2

   Behaviours.generateFileMl1m(countOfItems, countOfRepetitions)

   bDF:DataFrame = Behaviours.readFromFileMl1m()
   print(bDF.head(10))

#os.chdir("..")
#generateBehaviour()