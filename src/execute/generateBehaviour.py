#!/usr/bin/python3

import os
import random
import numpy as np

from datasets.ml.behaviours import Behaviours #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class


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

   # ML
#   Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_STATIC08, uBehavStatic08Desc)
#   Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_STATIC06, uBehavStatic06Desc)
#   Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_STATIC04, uBehavStatic04Desc)
#   Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_STATIC02, uBehavStatic02Desc)

#   Behaviours.generateFileMl1m(countOfItems, countOfRepetitions, Behaviours.BHVR_LINEAR0109, uBehavLinear0109Desc)

   # RR
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_STATIC08, uBehavStatic08Desc)
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_STATIC06, uBehavStatic06Desc)
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_STATIC04, uBehavStatic04Desc)
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_STATIC02, uBehavStatic02Desc)

#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_LINEAR0109, uBehavLinear0109Desc)

   # ST
#   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_LINEAR0109, uBehavLinear0109Desc)
   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_STATIC06, uBehavStatic06Desc)
   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_STATIC04, uBehavStatic04Desc)
   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_STATIC02, uBehavStatic02Desc)
   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_LINEAR0109, uBehavLinear0109Desc)


if __name__ == "__main__":
   os.chdir("..")
   generateBehaviour()