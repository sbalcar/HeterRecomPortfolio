#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class


def generateBehaviour():
   print("Generate Behaviour")
   os.chdir("..")

   np.random.seed(42)
   random.seed(42)

   Behaviours.generateFileMl1m(20)

   bDF:DataFrame = Behaviours.readFromFileMl1m()
   print(bDF.head(10))


generateBehaviour()