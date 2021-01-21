#!/usr/bin/python3

from typing import List

from input.inputRecomMLDefinition import InputRecomMLDefinition #class

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderVSKNN import RecommenderVMContextKNN #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from configuration.configuration import Configuration #class


class InputRecomSTDefinition:

    THE_MOST_POPULAR:str = InputRecomMLDefinition.THE_MOST_POPULAR
    KNN:str = InputRecomMLDefinition.KNN
    VMC_KNN:str = "vmContextKNN"

    COS_CB_OHE_MEAN1:str = "cosCBoneMean1"   # the best
    COS_CB_OHE_WEIGHTEDMEAN5:str = "cosCBoneWeightedMean5"  # the second best

    W2V_ALL100000WS1VS32_MAX1:str = "w2vtalli100000ws1vs32upsmaxups1"   # the best
    W2V_ALL200000WS1VS64_WEIGHTEDMEAN5:str = "w2vtalli200000ws1vs64upsweightedMeanups5"  # the second best

    BPRMF_F50I20LR01R003:str = "bpmMFf50i20lr01r003"   # the best
    BPRMF_F50I20LR01R001:str = "bpmMFf20i50lr01r001"   # the second best


    @staticmethod
    def exportRDescTheMostPopular():
        return RecommenderDescription(RecommenderTheMostPopular,
                {})

    @staticmethod
    def exportRDescKNN():
        return RecommenderDescription(RecommenderItemBasedKNN,
                {})

    @staticmethod
    def exportRDescVMContextKNN():
        return RecommenderDescription(RecommenderVMContextKNN, {
            RecommenderVMContextKNN.ARG_K: 50})

    @staticmethod
    def exportRDescCosineCBcbdOHEupsmeanups1():
        return RecommenderDescription(RecommenderCosineCB, {
                RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbSTDataFileWithPathOHE,
                RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY: "mean",
                RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 1})

    @staticmethod
    def exportRDescCosineCBcbdOHEupsweightedMeanups5():
        return RecommenderDescription(RecommenderCosineCB, {
                RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbSTDataFileWithPathOHE,
                RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY: "weightedMean",
                RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 5})

    @staticmethod
    def exportRDescW2Vtalli100000ws1vs32upsmaxups1():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_ITERATIONS: 100000,
                RecommenderW2V.ARG_TRAIN_VARIANT: "all",
                RecommenderW2V.ARG_USER_PROFILE_SIZE: 1,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY: "max",
                RecommenderW2V.ARG_VECTOR_SIZE: 32,
                RecommenderW2V.ARG_WINDOW_SIZE: 1})

    @staticmethod
    def exportRDescW2talli200000ws1vs64upsweightedMeanups5():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_ITERATIONS: 200000,
                RecommenderW2V.ARG_TRAIN_VARIANT: "all",
                RecommenderW2V.ARG_USER_PROFILE_SIZE: 5,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY: "weightedMean",
                RecommenderW2V.ARG_VECTOR_SIZE: 64,
                RecommenderW2V.ARG_WINDOW_SIZE: 1})

    @staticmethod
    def exportRDescBPRMFf50i20lr01r003():
        return RecommenderDescription(RecommenderBPRMF, {
                RecommenderBPRMF.ARG_FACTORS: 20,
                RecommenderBPRMF.ARG_ITERATIONS: 50,
                RecommenderBPRMF.ARG_LEARNINGRATE: 0.1,
                RecommenderBPRMF.ARG_REGULARIZATION: 0.03})

    @staticmethod
    def exportRDescBPRMFf20i50lr01r001():
        return RecommenderDescription(RecommenderBPRMF, {
                RecommenderBPRMF.ARG_FACTORS: 20,
                RecommenderBPRMF.ARG_ITERATIONS: 50,
                RecommenderBPRMF.ARG_LEARNINGRATE: 0.1,
                RecommenderBPRMF.ARG_REGULARIZATION: 0.01})


    @classmethod
    def exportPairOfRecomIdsAndRecomDescrs(cls):

        recom:str = "Recom"

        rIDsPop:List[str] = [recom + cls.THE_MOST_POPULAR.title()]
        rDescsPop:List[RecommenderDescription] = [cls.exportRDescTheMostPopular()]

        rIDsKNN:List[str] = [recom + cls.KNN.title()]
        rDescsKNN:List[RecommenderDescription] = [cls.exportRDescKNN()]

        rIDsVMCKNN:List[str] = [recom + cls.VMC_KNN.title()]
        rDescsVMCKNN:List[RecommenderDescription] = [cls.exportRDescVMContextKNN()]
        #rIDsVMCKNN:List[str] = []
        #rDescsVMCKNN:List[RecommenderDescription] = []


        rIDsBPRMF:List[str] = [recom + cls.BPRMF_F50I20LR01R003.title(), recom + cls.BPRMF_F50I20LR01R001.title()]
        rDescsBPRMF:List[RecommenderDescription] = [cls.exportRDescBPRMFf50i20lr01r003(), cls.exportRDescBPRMFf20i50lr01r001()]

        rIDsCOSCB:List[str] = [recom + cls.COS_CB_OHE_MEAN1.title(), recom + cls.COS_CB_OHE_WEIGHTEDMEAN5.title()]
        rDescsCOSCB:List[RecommenderDescription] = [cls.exportRDescCosineCBcbdOHEupsmeanups1(), cls.exportRDescCosineCBcbdOHEupsweightedMeanups5()]

        rIDsW2V:List[str] = [recom + cls.W2V_ALL100000WS1VS32_MAX1.title(), recom + cls.W2V_ALL200000WS1VS64_WEIGHTEDMEAN5.title()]
        rDescsW2V:List[RecommenderDescription] = [cls.exportRDescW2Vtalli100000ws1vs32upsmaxups1(), cls.exportRDescW2talli200000ws1vs64upsweightedMeanups5()]
        #rIDsW2V: List[str] = []
        #rDescsW2V: List[RecommenderDescription] = []

        rIDs:List[str] = rIDsPop + rIDsKNN + rIDsVMCKNN + rIDsBPRMF + rIDsCOSCB + rIDsW2V
        rDescs:List[RecommenderDescription] = rDescsPop + rDescsKNN + rDescsVMCKNN + rDescsBPRMF + rDescsCOSCB + rDescsW2V

        return (rIDs, rDescs)


    @classmethod
    def exportPairOfRecomIdsAndRecomDescrsRetailRocket(cls):

        recom:str = "Recom"

        rIDs:List[str] = [recom + InputRecomMLDefinition.THE_MOST_POPULAR.title()]
        rDescs:List[RecommenderDescription] = [InputRecomMLDefinition.exportRDescTheMostPopular()]

        return (rIDs, rDescs)


    @staticmethod
    def exportInputRecomDefinition(cls, recommenderID:str):
        if recommenderID == cls.THE_MOST_POPULAR:
            return cls.exportRDescTheMostPopular()
        elif recommenderID == cls.KNN:
            return cls.exportRDescKNN()
        elif recommenderID == cls.VMC_KNN:
            return cls.exportRDescVMContextKNNk25()
        elif recommenderID == cls.COS_CB_OHE_MEAN1:
            return cls.exportRDescCosineCBcbdOHEupsmeanups1()
        elif recommenderID == cls.COS_CB_OHE_WEIGHTEDMEAN5:
            return cls.exportRDescCosineCBcbdOHEupsweightedMeanups5()
        elif recommenderID == cls.W2V_ALL100000WS1VS32_MAX1:
            return cls.exportRDescW2Vtalli100000ws1vs32upsmaxups1()
        elif recommenderID == cls.W2V_ALL200000WS1VS64_WEIGHTEDMEAN5:
            return cls.exportRDescW2talli200000ws1vs64upsweightedMeanups5()
        elif recommenderID == cls.BPRMF_F50I20LR01R003:
            return cls.exportRDescBPRMFf50i20lr01r003()
        elif recommenderID == cls.BPRMF_F50I20LR01R001:
            return cls.exportRDescBPRMFf20i50lr01r001()

