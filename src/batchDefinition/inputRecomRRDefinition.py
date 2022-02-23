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
    THE_MOST_POPULAR:str = "TheMostPopular"
    KNN:str = "KNN"
    BPRMF:str = "BPRMF"
    VMC_KNN:str = "VMContextKNN"

    COSINECB:str = "CosineCB"
    W2V:str = "W2V"

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
        return RecommenderDescription(RecommenderItemBasedKNN, {
                RecommenderItemBasedKNN.ARG_K:25,
                RecommenderItemBasedKNN.ARG_UPDATE_THRESHOLD:500})


    @staticmethod
    def exportRDescBPRMF():
        return RecommenderDescription(RecommenderBPRMF, {
                RecommenderBPRMF.ARG_FACTORS: 100,
                RecommenderBPRMF.ARG_ITERATIONS: 50,
                RecommenderBPRMF.ARG_LEARNINGRATE: 0.1,
                RecommenderBPRMF.ARG_REGULARIZATION: 0.01})


    @staticmethod
    def exportRDescVMContextKNN():
        return RecommenderDescription(RecommenderVMContextKNN,{
                RecommenderVMContextKNN.ARG_K:50})

    @staticmethod
    def exportRDescCosineCB():
        return RecommenderDescription(RecommenderCosineCB, {
                RecommenderCosineCB.ARG_CB_DATA_PATH:"../data/simMatrixRR.npz",
                RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 1,
                RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"max"})

    @staticmethod
    def exportRDescW2V():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_LEARNING_RATE: 0.6,
                RecommenderW2V.ARG_ITERATIONS: 100000,
                RecommenderW2V.ARG_TRAIN_VARIANT:"all",
                RecommenderW2V.ARG_USER_PROFILE_SIZE: 3,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"weightedMean",
                RecommenderW2V.ARG_VECTOR_SIZE: 64,
                RecommenderW2V.ARG_WINDOW_SIZE: 3})



    @classmethod
    def exportPairOfRecomIdsAndRecomDescrs(cls):

        recom:str = "Recom"

        rIDsPop:List[str] = [recom + cls.THE_MOST_POPULAR.title()]
        rDescsPop:List[RecommenderDescription] = [cls.exportRDescTheMostPopular()]

        rIDsKNN:List[str] = [recom + cls.KNN.title()]
        rDescsKNN:List[RecommenderDescription] = [cls.exportRDescKNN()]

        rIDsVMCKNN:List[str] = [recom + cls.VMC_KNN.title()]
        rDescsVMCKNN:List[RecommenderDescription] = [cls.exportRDescVMContextKNN()]

        rIDsBPRMF:List[str] = [recom + cls.BPRMF.title()]
        rDescsBPRMF:List[RecommenderDescription] = [cls.exportRDescBPRMF()]

        rIDsW2V:List[str] = [recom + cls.W2V.title()]
        rDescsW2V:List[RecommenderDescription] = [cls.exportRDescW2V()]

        rIDsCB:List[str] = [recom + cls.COSINECB.title()]
        rDescsCB:List[RecommenderDescription] = [cls.exportRDescCosineCB()]

        rIDs:List[str] = rIDsPop + rIDsKNN + rIDsVMCKNN + rIDsBPRMF + rIDsW2V + rIDsCB
        rDescs:List[RecommenderDescription] = rDescsPop + rDescsKNN + rDescsVMCKNN + rDescsBPRMF + rDescsW2V + rDescsCB

        return (rIDs, rDescs)


    @staticmethod
    def exportInputRecomDefinition(recommenderID:str):
        if recommenderID == InputRecomRRDefinition.THE_MOST_POPULAR:
            return InputRecomRRDefinition.exportRDescTheMostPopular()
        elif recommenderID == InputRecomRRDefinition.KNN:
            return InputRecomRRDefinition.exportRDescKNN()
        elif recommenderID == InputRecomRRDefinition.VMC_KNN:
            return InputRecomRRDefinition.exportRDescVMContextKNN()
        elif recommenderID == InputRecomRRDefinition.BPRMF:
            return InputRecomRRDefinition.exportRDescBPRMF()
        elif recommenderID == InputRecomRRDefinition.W2V:
            return InputRecomRRDefinition.exportRDescW2V()
        elif recommenderID == InputRecomRRDefinition.COSINECB:
            return InputRecomRRDefinition.exportRDescCosineCB()
