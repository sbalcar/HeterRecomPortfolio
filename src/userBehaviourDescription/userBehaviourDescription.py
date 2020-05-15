#!/usr/bin/python3

from typing import List


def observationalStaticProbabilityFnc(probability: float, numberOfItems: int):
    if probability < 0.0 or probability > 1.0:
        raise ValueError("Argument probability not in range <0.0, 1.0>.")
    if numberOfItems < 0:
        raise ValueError("Argument numberOfItems can't contain negative value.")
    return [probability] * numberOfItems


def observationalLinearProbabilityFnc(minProbability: float, maxProbability: float, numberOfItems:int):
    if minProbability < 0.0 or minProbability > 1.0:
        raise ValueError("Argument minProbability not in range <0.0, 1.0>.")
    if maxProbability < 0.0 or maxProbability > 1.0:
        raise ValueError("Argument maxProbability not in range <0.0, 1.0>.")
    if minProbability >= maxProbability:
        raise ValueError("Argument minProbability isn't lesser than maxProbability.")
    if numberOfItems < 0:
        raise ValueError("Argument numberOfItems can't contain negative value.")

    diff:float = (maxProbability - minProbability) / (numberOfItems - 1)
    probabilities:List[float] = [minProbability + indexI * diff for indexI in range(numberOfItems)]
    probabilities.reverse()

    return probabilities

class UserBehaviourDescription:

    def __init__(self, uObservationalProbFnc, arguments:List):

        self._uObservationalProbFnc = uObservationalProbFnc
        self._arguments = arguments


    def getProbabilityOfBehavior(self, numberOfItems:int):

        return self._uObservationalProbFnc(*self._arguments, numberOfItems)