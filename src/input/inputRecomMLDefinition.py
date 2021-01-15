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


class InputRecomMLDefinition:

    # ML methods
    COS_CB_MEAN:str = "cosCBmean"
    COS_CB_WINDOW3:str = "cosCBwindow3"
    THE_MOST_POPULAR:str = "theMostPopular"
    W2V_POSNEG_MEAN:str = "w2vPosnegMean"
    W2V_POSNEG_WINDOW3:str = "w2vPosnegWindow3"

    KNN:str = "KNN"
    VMC_KNN:str = "vmContextKNN"

    BPRMF:str = "BPRMF"

    @staticmethod
    def exportRDescTheMostPopular():
        return RecommenderDescription(RecommenderTheMostPopular,
                {})


    @staticmethod
    def exportRDescCBmean():
        return RecommenderDescription(RecommenderCosineCB, {
                RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbML1MDataFileWithPathTFIDF,
#            RecommenderCosineCB.ARG_CB_DATA_PATH: Configuration.cbSTDataFileWithPathTFIDF,
            RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 5,
                RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"mean"})

    @staticmethod
    def exportRDescCBwindow3():
        return RecommenderDescription(RecommenderCosineCB, {
                RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbML1MDataFileWithPathTFIDF,
                RecommenderCosineCB.ARG_USER_PROFILE_SIZE: 5,
                RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"window3"})


    @staticmethod
    def exportRDescW2vPositiveMax():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_ITERATIONS: 50000,
                RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"max",
                RecommenderW2V.ARG_VECTOR_SIZE: 128,
                RecommenderW2V.ARG_WINDOW_SIZE: 5})


    @staticmethod
    def exportRDescW2vPositiveWindow10():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_ITERATIONS: 50000,
                RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window10",
                RecommenderW2V.ARG_VECTOR_SIZE: 128,
                RecommenderW2V.ARG_WINDOW_SIZE: 5})

    @staticmethod
    def exportRDescW2vPosnegMean():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_ITERATIONS: 50000,
                RecommenderW2V.ARG_TRAIN_VARIANT:"posneg",
                RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"mean",
                RecommenderW2V.ARG_VECTOR_SIZE: 128,
                RecommenderW2V.ARG_WINDOW_SIZE: 5})
    @staticmethod
    def exportRDescW2vPosnegWindow3():
        return RecommenderDescription(RecommenderW2V, {
                RecommenderW2V.ARG_ITERATIONS: 50000,
                RecommenderW2V.ARG_TRAIN_VARIANT:"posneg",
                RecommenderW2V.ARG_USER_PROFILE_SIZE: -1,
                RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window3",
                RecommenderW2V.ARG_VECTOR_SIZE: 128,
                RecommenderW2V.ARG_WINDOW_SIZE: 5})

    @staticmethod
    def exportRDescKNN():
        return RecommenderDescription(RecommenderItemBasedKNN,
                {})

    @staticmethod
    def exportRDescVMContextKNN():
        return RecommenderDescription(RecommenderVMContextKNN, {
            RecommenderVMContextKNN.ARG_K: 50})

    @staticmethod
    def exportRDescBPRMF():
        return RecommenderDescription(RecommenderBPRMF, {
                RecommenderBPRMF.ARG_FACTORS: 20,
                RecommenderBPRMF.ARG_ITERATIONS: 50,
                RecommenderBPRMF.ARG_LEARNINGRATE: 0.003,
                RecommenderBPRMF.ARG_REGULARIZATION: 0.003})


    @classmethod
    def exportPairOfRecomIdsAndRecomDescrs(cls):

        recom:str = "Recom"

        rIDsCB:List[str] = [recom + cls.COS_CB_MEAN.title(), recom + cls.COS_CB_WINDOW3.title()]
        rDescsCB:List[RecommenderDescription] = [cls.exportRDescCBmean(), cls.exportRDescCBwindow3()]

        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPositiveWindow10", "RecomW2vPosnegMean", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPositiveMax, rDescW2vPositiveWindow10, rDescW2vPosnegMean, rDescW2vPosnegWindow10]
        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPosnegMax", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPosnegMax, rDescW2vPosnegMax, rDescW2vPosnegWindow10]

        #rIDsW2V:List[str] = [recom + cls.W2V_POSNEG_MEAN.title(), recom + cls.W2V_POSNEG_WINDOW3.title()]
        #rDescsW2V:List[RecommenderDescription] = [cls.exportRDescW2vPosnegMean(), cls.exportRDescW2vPosnegWindow3()]
        rIDsW2V:List[str] = []
        rDescsW2V:List[RecommenderDescription] = []

        rIDsKNN:List[str] = [recom + cls.KNN.title()]
        rDescsKNN:List[RecommenderDescription] = [cls.exportRDescKNN()]

        #rIDsVMCKNN:List[str] = [recom + cls.VMC_KNN.title()]
        #rDescsVMCKNN:List[RecommenderDescription] = [cls.exportRDescVMContextKNN()]
        rIDsVMCKNN:List[str] = []
        rDescsVMCKNN:List[RecommenderDescription] = []

        rIDsBPRMF:List[str] = [recom + cls.BPRMF.title()]
        rDescsBPRMF:List[RecommenderDescription] = [cls.exportRDescBPRMF()]

        rIDsPop:List[str] = [recom + cls.THE_MOST_POPULAR.title()]
        rDescsPop:List[RecommenderDescription] = [cls.exportRDescTheMostPopular()]

        rIDs:List[str] = rIDsCB + rIDsW2V + rIDsKNN + rIDsVMCKNN + rIDsBPRMF + rIDsPop
        rDescs:List[RecommenderDescription] = rDescsCB + rDescsW2V + rDescsKNN + rDescsVMCKNN + rDescsBPRMF + rDescsPop

        return (rIDs, rDescs)




    @classmethod
    def exportPairOfRecomIdsAndRecomDescrsRetailRocket(cls):

        recom:str = "Recom"

        rIDs:List[str] = [recom + InputRecomMLDefinition.THE_MOST_POPULAR.title()]
        rDescs:List[RecommenderDescription] = [InputRecomMLDefinition.exportRDescTheMostPopular()]

        return (rIDs, rDescs)



    @staticmethod
    def exportInputRecomDefinition(recommenderID:str):
        if recommenderID == InputRecomMLDefinition.COS_CB_MEAN:
            return InputRecomMLDefinition.exportRDescCBmean()
        elif recommenderID == InputRecomMLDefinition.COS_CB_WINDOW3:
            return InputRecomMLDefinition.exportRDescCBwindow3()
        elif recommenderID == InputRecomMLDefinition.THE_MOST_POPULAR:
            return InputRecomMLDefinition.exportRDescTheMostPopular()
        elif recommenderID == InputRecomMLDefinition.W2V_POSNEG_MEAN:
            return InputRecomMLDefinition.exportRDescW2vPosnegMean()
        elif recommenderID == InputRecomMLDefinition.W2V_POSNEG_WINDOW3:
            return InputRecomMLDefinition.exportRDescW2vPosnegWindow3()
        elif recommenderID == InputRecomMLDefinition.KNN:
            return InputRecomMLDefinition.exportRDescKNN()
        elif recommenderID == InputRecomMLDefinition.VMC_KNN:
            return InputRecomMLDefinition.exportRDescVMContextKNN()
        elif recommenderID == InputRecomMLDefinition.BPRMF:
            return InputRecomMLDefinition.exportRDescBPRMF()