#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderVSKNN import RecommenderVMContextKNN #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from configuration.configuration import Configuration #class


class InputRecomRRDefinition:

    # RR methods
    THE_MOST_POPULAR :str = "TheMostPopular"
    KNN :str = "KNN"
    #VMC_KNN25 :str = "VMContextKNN25"

    #W2VweightedMeanups3 :str = "W2Vtpositivei50000ws1vs32upsweightedMeanups3"
    #W2VweightedMeanups7 :str = "W2Vtpositivei50000ws1vs64upsweightedMeanups7"

    #COSINECBcbdOHEupsweightedMeanups3 :str = "CosineCBcbdOHEupsweightedMeanups3"
    #COSINECBcbdOHEupsmaxups1 :str = "CosineCBcbdOHEupsmaxups1"

    #BPRMFf100i10lr0003r01 :str = "BPRMFf100i10lr0003r01"
    #BPRMFf20i20lr0003r01 :str = "BPRMFf20i20lr0003r01"


    @staticmethod
    def exportRDescTheMostPopular():
        return RecommenderDescription(RecommenderTheMostPopular,
                                      {})

    @staticmethod
    def exportRDescKNN():
        return RecommenderDescription(RecommenderItemBasedKNN,
                                      {})
