#!/usr/bin/python3

import os
import random
import numpy as np

from typing import List

from pandas.core.frame import DataFrame #class

from datasets.behaviours import Behaviours #class

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function


def generateBehaviour():
   print("Generate Behaviours")

   np.random.seed(42)
   random.seed(42)

   countOfItems:int = 20
   countOfRepetitions:int = 5

   uBehavStatic08Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.8])
   uBehavStatic06Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.6])
   uBehavStatic04Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.4])
   uBehavStatic02Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.2])

   uBehavLinear0109Desc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1, 0.9])


   Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_STATIC08, uBehavStatic08Desc)
   #Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_STATIC06, uBehavStatic06Desc)
   #Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_STATIC04, uBehavStatic04Desc)
   #Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_STATIC02, uBehavStatic02Desc)

   Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_LINEAR0109, uBehavLinear0109Desc)


   #behaviourFile:str = Behaviours.getFile(Behaviours.BHVR_STATIC08)
   #behaviourDF:DataFrame = Behaviours.readFromFileMl1m(behaviourFile)
   #print(behaviourDF.head(10))




if __name__ == "__main__":
   os.chdir("..")
   generateBehaviour()