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


    @classmethod
    def exportPairOfRecomIdsAndRecomDescrs(cls):

        recom:str = "Recom"

        rIDsPop:List[str] = [recom + cls.THE_MOST_POPULAR.title()]
        rDescsPop:List[RecommenderDescription] = [cls.exportRDescTheMostPopular()]

        rIDsKNN:List[str] = [recom + cls.KNN.title()]
        rDescsKNN:List[RecommenderDescription] = [cls.exportRDescKNN()]

#        rIDsVMCKNN:List[str] = [recom + cls.VMC_KNN25.title()]
#        rDescsVMCKNN:List[RecommenderDescription] = [cls.exportRDescVMContextKNNk25()]
        #rIDsVMCKNN:List[str] = []
        #rDescsVMCKNN:List[RecommenderDescription] = []

#        rIDsW2V:List[str] = [recom + cls.W2VweightedMeanups3.title(), recom + cls.W2VweightedMeanups7.title()]
#        rDescsW2V:List[RecommenderDescription] = [cls.exportRDescW2Vtpositivei50000ws1vs32upsweightedMeanups3(), cls.exportRDescW2Vtpositivei50000ws1vs64upsweightedMeanups7()]
        #rIDsW2V: List[str] = []
        #rDescsW2V: List[RecommenderDescription] = []

#        rIDsCB:List[str] = [recom + cls.COSINECBcbdOHEupsweightedMeanups3.title(), recom + cls.COSINECBcbdOHEupsmaxups1.title()]
#        rDescsCB:List[RecommenderDescription] = [cls.exportRDescCosineCBcbdOHEupsweightedMeanups3(), cls.exportRDescCosineCBcbdOHEupsmaxups1()]

#        rIDsBPRMF:List[str] = [recom + cls.BPRMFf100i10lr0003r01.title(), recom + cls.BPRMFf20i20lr0003r01.title()]
#        rDescsBPRMF:List[RecommenderDescription] = [cls.exportRDescBPRMFf100i10lr0003r01(), cls.exportRDescBPRMFf20i20lr0003r01()]

#        rIDs:List[str] = rIDsPop + rIDsKNN + rIDsVMCKNN + rIDsW2V + rIDsCB + rIDsBPRMF
#        rDescs:List[RecommenderDescription] = rDescsPop + rDescsKNN + rDescsVMCKNN + rDescsW2V + rDescsCB + rDescsBPRMF

        rIDs:List[str] = rIDsPop + rIDsKNN
        rDescs:List[RecommenderDescription] = rDescsPop + rDescsKNN

        return (rIDs, rDescs)


    @staticmethod
    def exportInputRecomDefinition(recommenderID:str):
        if recommenderID == InputRecomRRDefinition.THE_MOST_POPULAR:
            return InputRecomRRDefinition.exportRDescTheMostPopular()
        elif recommenderID == InputRecomRRDefinition.KNN:
            return InputRecomRRDefinition.exportRDescKNN()
