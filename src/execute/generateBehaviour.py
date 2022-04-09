#!/usr/bin/python3

import os
import random
import numpy as np

from datasets.ml.behavioursML import BehavioursML #class
from datasets.retailrocket.behavioursRR import BehavioursRR #class
from datasets.slantour.behavioursST import BehavioursST #class


from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class
from userBehaviourDescription.userBehaviourDescription import observationalStaticProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalLinearProbabilityFnc #function
from userBehaviourDescription.userBehaviourDescription import observationalPowerLawFnc #function

def generateBehaviour():
   print("Generate Behaviours")

   np.random.seed(42)
   random.seed(42)

   countOfItems:int = 100
   countOfRepetitions:int = 5

   uBehavStatic08Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.8])
   uBehavStatic06Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.6])
   uBehavStatic04Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.4])
   uBehavStatic02Desc:UserBehaviourDescription = UserBehaviourDescription(observationalStaticProbabilityFnc, [0.2])

   uBehavLinear0109Desc:UserBehaviourDescription = UserBehaviourDescription(observationalLinearProbabilityFnc, [0.1, 0.9])

   uBehavPowerlaw054min048:UserBehaviourDescription = UserBehaviourDescription(observationalPowerLawFnc, [0.54, -0.48])


# ML
#   BehavioursML.generateFileMl1m(countOfItems, countOfRepetitions, BehavioursML.BHVR_STATIC08, uBehavStatic08Desc)
#   BehavioursML.generateFileMl1m(countOfItems, countOfRepetitions, BehavioursML.BHVR_STATIC06, uBehavStatic06Desc)
#   BehavioursML.generateFileMl1m(countOfItems, countOfRepetitions, BehavioursML.BHVR_STATIC04, uBehavStatic04Desc)
   BehavioursML.generateFileMl1m(countOfItems, countOfRepetitions, BehavioursML.BHVR_STATIC02, uBehavStatic02Desc)

#   BehavioursML.generateFileMl1m(countOfItems, countOfRepetitions, BehavioursML.BHVR_LINEAR0109, uBehavLinear0109Desc)
#   BehavioursML.generateFileMl1m(countOfItems, countOfRepetitions, BehavioursML.BHVR_POWERLAW054MIN048, uBehavPowerlaw054min048)

   # RR
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_STATIC08, uBehavStatic08Desc)
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_STATIC06, uBehavStatic06Desc)
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_STATIC04, uBehavStatic04Desc)
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_STATIC02, uBehavStatic02Desc)

#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_LINEAR0109, uBehavLinear0109Desc)
#   BehavioursRR.generateFileRR(countOfItems, countOfRepetitions, BehavioursRR.BHVR_POWERLAW054MIN048, uBehavPowerlaw054min048)


   # ST
#   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_LINEAR0109, uBehavLinear0109Desc)
#   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_STATIC08, uBehavStatic08Desc)
#   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_STATIC06, uBehavStatic06Desc)
#   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_STATIC04, uBehavStatic04Desc)
#   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_STATIC02, uBehavStatic02Desc)

#   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_LINEAR0109, uBehavLinear0109Desc)
#   BehavioursST.generateFileST(countOfItems, countOfRepetitions, BehavioursST.BHVR_POWERLAW054MIN048, uBehavPowerlaw054min048)


if __name__ == "__main__":
   os.chdir("..")
   generateBehaviour()