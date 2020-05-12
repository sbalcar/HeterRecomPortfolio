#!/usr/bin/python3

from typing import List

import numpy as np

from userBehaviourDescription.userBehaviourDescription import UserBehaviourDescription #class


class UserBehaviourSimulator:

    def simulate(self, uBehaviourDesc:UserBehaviourDescription, numberOfItems:int):

        probabilitiesOfObservational:List[float] = uBehaviourDesc.getProbabilityOfBehavior(numberOfItems)
        #print("probabilitiesOfObservational: " + str(probabilitiesOfObservational))

        probabilitiesGenerated:List[float] =  np.random.uniform(low=0.0, high=1.0, size=numberOfItems)
        #print("probabilitiesGenerated: " + str(probabilitiesGenerated))

        behaviour:List[bool] = list(map(lambda x, y: x > y, probabilitiesOfObservational, probabilitiesGenerated))
        #print(behaviour)

        return behaviour
