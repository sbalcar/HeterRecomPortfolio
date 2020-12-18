#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class
from recommender.recommenderItemBasedKNN import RecommenderItemBasedKNN #class
from recommender.recommenderBPRMF import RecommenderBPRMF #class

from recommender.recommenderTheMostSold import RecommenderTheMostSold #class

from configuration.configuration import Configuration #class


class InputRecomDefinition:

    # ML methods
    COS_CB_MEAN:str = "cosCBmean"
    COS_CB_WINDOW3:str = "cosCBwindow3"
    THE_MOST_POPULAR:str = "theMostPopular"
    W2V_POSNEG_MEAN:str = "w2vPosnegMean"
    W2V_POSNEG_WINDOW3:str = "w2vPosnegWindow3"
    KNN:str = "KNN"
    BPRMF:str = "BPRMF"

    # batchesRetailrocket methods
    THE_MOST_SOLD:str = "theMostSold"

    @staticmethod
    def exportRDescTheMostPopular(datasetID:str):
        return RecommenderDescription(RecommenderTheMostPopular,
                {})


    @staticmethod
    def exportRDescCBmean(datasetID:str):
        return RecommenderDescription(RecommenderCosineCB,
                {RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbDataFileWithPathTFIDF,
                 RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"mean"})
    @staticmethod
    def exportRDescCBwindow3(datasetID:str):
        return RecommenderDescription(RecommenderCosineCB,
                {RecommenderCosineCB.ARG_CB_DATA_PATH:Configuration.cbDataFileWithPathTFIDF,
                 RecommenderCosineCB.ARG_USER_PROFILE_STRATEGY:"window3"})


    @staticmethod
    def exportRDescW2vPositiveMax(datasetID:str):
        return RecommenderDescription(RecommenderW2V,
                {RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                 RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"max",
                 RecommenderW2V.ARG_DATASET_ID:datasetID})
    @staticmethod
    def exportRDescW2vPositiveWindow10(datasetID:str):
        return RecommenderDescription(RecommenderW2V,
                {RecommenderW2V.ARG_TRAIN_VARIANT:"positive",
                 RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window10",
                 RecommenderW2V.ARG_DATASET_ID:datasetID})
    @staticmethod
    def exportRDescW2vPosnegMean(datasetID:str):
        return RecommenderDescription(RecommenderW2V,
                {RecommenderW2V.ARG_TRAIN_VARIANT:"posneg",
                 RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"mean",
                 RecommenderW2V.ARG_DATASET_ID:datasetID})
    @staticmethod
    def exportRDescW2vPosnegWindow3(datasetID:str):
        return RecommenderDescription(RecommenderW2V,
                {RecommenderW2V.ARG_TRAIN_VARIANT:"posneg",
                 RecommenderW2V.ARG_USER_PROFILE_STRATEGY:"window3",
                 RecommenderW2V.ARG_DATASET_ID:datasetID})

    @staticmethod
    def exportRDescKNN(datasetID:str):
        return RecommenderDescription(RecommenderItemBasedKNN,
                {})

    @staticmethod
    def exportRDescBPRMF(datasetID:str):
        return RecommenderDescription(RecommenderBPRMF,
                {})


    @staticmethod
    def exportPairOfRecomIdsAndRecomDescrsML(datasetID:str):

        recom:str = "Recom"

        rIDsCB:List[str] = [recom + InputRecomDefinition.COS_CB_MEAN.title(), recom + InputRecomDefinition.COS_CB_WINDOW3.title()]
        rDescsCB:List[RecommenderDescription] = [InputRecomDefinition.exportRDescCBmean(datasetID), InputRecomDefinition.exportRDescCBwindow3(datasetID)]

        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPositiveWindow10", "RecomW2vPosnegMean", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPositiveMax, rDescW2vPositiveWindow10, rDescW2vPosnegMean, rDescW2vPosnegWindow10]
        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPosnegMax", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPosnegMax, rDescW2vPosnegMax, rDescW2vPosnegWindow10]

        rIDsW2V:List[str] = [recom + InputRecomDefinition.W2V_POSNEG_MEAN.title(), recom + InputRecomDefinition.W2V_POSNEG_WINDOW3.title()]
        rDescsW2V:List[RecommenderDescription] = [InputRecomDefinition.exportRDescW2vPosnegMean(datasetID), InputRecomDefinition.exportRDescW2vPosnegWindow3(datasetID)]

        rIDsKNN:List[str] = [recom + InputRecomDefinition.KNN.title()]
        rDescsKNN:List[RecommenderDescription] = [InputRecomDefinition.exportRDescKNN(datasetID)]

        rIDsBPRMF:List[str] = [recom + InputRecomDefinition.BPRMF.title()]
        rDescsBPRMF:List[RecommenderDescription] = [InputRecomDefinition.exportRDesBPRMF(datasetID)]

        rIDsPop:List[str] = [recom + InputRecomDefinition.THE_MOST_POPULAR.title()]
        rDescsPop:List[RecommenderDescription] = [InputRecomDefinition.exportRDescTheMostPopular(datasetID)]

        rIDs:List[str] = rIDsCB + rIDsW2V + rIDsKNN + rIDsBPRMF + rIDsPop
        rDescs:List[RecommenderDescription] = rDescsCB + rDescsW2V + rDescsKNN + rDescsBPRMF + rDescsPop

        return (rIDs, rDescs)



    @staticmethod
    def exportRDescTheMostSold(datasetID:str):
        return RecommenderDescription(RecommenderTheMostSold,
                {})


    @staticmethod
    def exportPairOfRecomIdsAndRecomDescrsRetailRocket(datasetID:str):

        recom:str = "Recom"

        rIDs:List[str] = [recom + InputRecomDefinition.THE_MOST_SOLD.title()]
        rDescs:List[RecommenderDescription] = [InputRecomDefinition.exportRDescTheMostSold(datasetID)]

        return (rIDs, rDescs)



    @staticmethod
    def exportInputRecomDefinition(recommenderID:str, datasetID:str):
        if recommenderID == InputRecomDefinition.COS_CB_MEAN:
            return InputRecomDefinition.exportRDescCBmean(datasetID)
        elif recommenderID == InputRecomDefinition.COS_CB_WINDOW3:
            return InputRecomDefinition.exportRDescCBwindow3(datasetID)
        elif recommenderID == InputRecomDefinition.THE_MOST_POPULAR:
            return InputRecomDefinition.exportRDescTheMostPopular(datasetID)
        elif recommenderID == InputRecomDefinition.W2V_POSNEG_MEAN:
            return InputRecomDefinition.exportRDescW2vPosnegMean(datasetID)
        elif recommenderID == InputRecomDefinition.W2V_POSNEG_WINDOW3:
            return InputRecomDefinition.exportRDescW2vPosnegWindow3(datasetID)
        elif recommenderID == InputRecomDefinition.KNN:
            return InputRecomDefinition.exportRDescKNN(datasetID)
        elif recommenderID == InputRecomDefinition.BPRMF:
            return InputRecomDefinition.exportRDescBPRMF(datasetID)
        elif recommenderID == InputRecomDefinition.THE_MOST_SOLD:
            return InputRecomDefinition.exportRDescTheMostSold(datasetID)
