#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderVSKNN import RecommenderVMContextKNN #class
from recommender.recommenderBPRMFImplicit import RecommenderBPRMFImplicit #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from configuration.configuration import Configuration #class


class InputRecomMLDefinition:

    # ML methods
    THE_MOST_POPULAR:str = "TheMostPopular"
    KNN:str = "KNN"
    VMC_KNN25:str = "VMContextKNN25"

    W2VweightedMeanups3:str = "W2Vtpositivei50000ws1vs32upsweightedMeanups3"
    W2VweightedMeanups7:str = "W2Vtpositivei50000ws1vs64upsweightedMeanups7"

    COSINECBcbdOHEupsweightedMeanups3:str = "CosineCBcbdOHEupsweightedMeanups3"
    COSINECBcbdOHEupsmaxups1:str = "CosineCBcbdOHEupsmaxups1"

    BPRMFIMPLf100i10lr0003r01:str = "BPRMFIMPLf100i10lr0003r01"
    BPRMFIMPLf20i20lr0003r01:str = "BPRMFIMPLf20i20lr0003r01"

    BPRMF:str = "BPRMF"

    @staticmethod
    def exportRDescTheMostPopular():
        return RecommenderDescription(RecommenderTheMostPopular,
                {})

    @staticmethod
    def exportRDescKNN():
        return RecommenderDescription(RecommenderItemBasedKNN, {
            RecommenderItemBasedKNN.ARG_K: 25,
            RecommenderItemBasedKNN.ARG_UPDATE_THRESHOLD: 500})

    @staticmethod
    def exportRDescVMContextKNNk25():
        return RecommenderDescription(RecommenderVMContextKNN, {
            RecommenderVMContextKNN.ARG_K: 25})


    @staticmethod
    def exportRDescCosineCBcbdOHEupsweightedMeanups3():
        return RecommenderDescription(RecommenderCosineCB, {
                RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbML1MDataFileWithPathOHE,
                RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 3,
                RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"weightedMean"})

    @staticmethod
    def exportRDescCosineCBcbdOHEupsmaxups1():
        return RecommenderDescription(RecommenderCosineCB, {
                RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbML1MDataFileWithPathOHE,
                RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 1,
                RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"max"})


    @staticmethod
    def exportRDescW2Vtpositivei50000ws1vs32upsweightedMeanups3():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                RecommenderW2V.ARG_ITERATIONS: 50000,
                RecommenderW2V.ARG_LEARNING_RATE: 1.0,
                RecommenderW2V.ARG_USER_PROFILE_SIZE: 3,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"weightedMean",
                RecommenderW2V.ARG_VECTOR_SIZE: 32,
                RecommenderW2V.ARG_WINDOW_SIZE: 1})


    @staticmethod
    def exportRDescW2Vtpositivei50000ws1vs64upsweightedMeanups7():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                RecommenderW2V.ARG_ITERATIONS: 50000,
                RecommenderW2V.ARG_LEARNING_RATE: 1.0,
                RecommenderW2V.ARG_USER_PROFILE_SIZE: 7,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"weightedMean",
                RecommenderW2V.ARG_VECTOR_SIZE: 64,
                RecommenderW2V.ARG_WINDOW_SIZE: 1})

    @staticmethod
    def exportRDescBPRMFIMPLf100i10lr0003r01():
        return RecommenderDescription(RecommenderBPRMFImplicit, {
                RecommenderBPRMFImplicit.ARG_FACTORS: 100,
                RecommenderBPRMFImplicit.ARG_ITERATIONS: 10,
                RecommenderBPRMFImplicit.ARG_LEARNINGRATE: 0.003,
                RecommenderBPRMFImplicit.ARG_REGULARIZATION: 0.1})

    @staticmethod
    def exportRDescBPRMFIMPLf20i20lr0003r01():
        return RecommenderDescription(RecommenderBPRMFImplicit, {
                RecommenderBPRMFImplicit.ARG_FACTORS: 20,
                RecommenderBPRMFImplicit.ARG_ITERATIONS: 20,
                RecommenderBPRMFImplicit.ARG_LEARNINGRATE: 0.003,
                RecommenderBPRMFImplicit.ARG_REGULARIZATION: 0.1})

    @staticmethod
    def exportRDescBPRMF():
        return RecommenderDescription(RecommenderBPRMF, {
                RecommenderBPRMF.ARG_EPOCHS: 2,
                RecommenderBPRMF.ARG_FACTORS: 10,
                RecommenderBPRMF.ARG_LEARNINGRATE: 0.05,
                RecommenderBPRMF.ARG_UREGULARIZATION: 0.0025,
                RecommenderBPRMF.ARG_BREGULARIZATION: 0,
                RecommenderBPRMF.ARG_PIREGULARIZATION: 0.0025,
                RecommenderBPRMF.ARG_NIREGULARIZATION: 0.00025})


    @classmethod
    def exportPairOfRecomIdsAndRecomDescrs(cls):

        recom:str = "Recom"

        rIDsPop:List[str] = [recom + cls.THE_MOST_POPULAR.title()]
        rDescsPop:List[RecommenderDescription] = [cls.exportRDescTheMostPopular()]

        rIDsKNN:List[str] = [recom + cls.KNN.title()]
        rDescsKNN:List[RecommenderDescription] = [cls.exportRDescKNN()]

        rIDsVMCKNN:List[str] = [recom + cls.VMC_KNN25.title()]
        rDescsVMCKNN:List[RecommenderDescription] = [cls.exportRDescVMContextKNNk25()]
        #rIDsVMCKNN:List[str] = []
        #rDescsVMCKNN:List[RecommenderDescription] = []

        rIDsW2V:List[str] = [recom + cls.W2VweightedMeanups3.title(), recom + cls.W2VweightedMeanups7.title()]
        rDescsW2V:List[RecommenderDescription] = [cls.exportRDescW2Vtpositivei50000ws1vs32upsweightedMeanups3(), cls.exportRDescW2Vtpositivei50000ws1vs64upsweightedMeanups7()]
        #rIDsW2V: List[str] = []
        #rDescsW2V: List[RecommenderDescription] = []

        rIDsCB:List[str] = [recom + cls.COSINECBcbdOHEupsweightedMeanups3.title(), recom + cls.COSINECBcbdOHEupsmaxups1.title()]
        rDescsCB:List[RecommenderDescription] = [cls.exportRDescCosineCBcbdOHEupsweightedMeanups3(), cls.exportRDescCosineCBcbdOHEupsmaxups1()]

        #rIDsBPRMF:List[str] = [recom + cls.BPRMFIMPLf100i10lr0003r01.title(), recom + cls.BPRMFIMPLf20i20lr0003r01.title()]
        #rDescsBPRMF:List[RecommenderDescription] = [cls.exportRDescBPRMFIMPLf100i10lr0003r01(), cls.exportRDescBPRMFIMPLf20i20lr0003r01()]
        rIDsBPRMF:List[str] = [recom + cls.BPRMF.title()]
        rDescsBPRMF:List[RecommenderDescription] = [cls.exportRDescBPRMF()]


        rIDs:List[str] = rIDsPop + rIDsKNN + rIDsVMCKNN + rIDsW2V + rIDsCB + rIDsBPRMF
        rDescs:List[RecommenderDescription] = rDescsPop + rDescsKNN + rDescsVMCKNN + rDescsW2V + rDescsCB + rDescsBPRMF

        return (rIDs, rDescs)


    @staticmethod
    def exportInputRecomDefinition(recommenderID:str):
        if recommenderID == InputRecomMLDefinition.THE_MOST_POPULAR:
            return InputRecomMLDefinition.exportRDescTheMostPopular()
        elif recommenderID == InputRecomMLDefinition.KNN:
            return InputRecomMLDefinition.exportRDescKNN()
        elif recommenderID == InputRecomMLDefinition.VMC_KNN25:
            return InputRecomMLDefinition.exportRDescVMContextKNNk25()

        elif recommenderID == InputRecomMLDefinition.W2VweightedMeanups3:
            return InputRecomMLDefinition.exportRDescW2Vtpositivei50000ws1vs32upsweightedMeanups3()
        elif recommenderID == InputRecomMLDefinition.W2VweightedMeanups7:
            return InputRecomMLDefinition.exportRDescW2Vtpositivei50000ws1vs64upsweightedMeanups7()

        elif recommenderID == InputRecomMLDefinition.COSINECBcbdOHEupsweightedMeanups3:
            return InputRecomMLDefinition.exportRDescCosineCBcbdOHEupsweightedMeanups3()
        elif recommenderID == InputRecomMLDefinition.COSINECBcbdOHEupsmaxups1:
            return InputRecomMLDefinition.exportRDescCosineCBcbdOHEupsmaxups1()

        elif recommenderID == InputRecomMLDefinition.BPRMFIMPLf100i10lr0003r01:
            return InputRecomMLDefinition.exportRDescBPRMFIMPLf100i10lr0003r01()
        elif recommenderID == InputRecomMLDefinition.BPRMFIMPLf20i20lr0003r01:
            return InputRecomMLDefinition.exportRDescBPRMFIMPLf20i20lr0003r01()
