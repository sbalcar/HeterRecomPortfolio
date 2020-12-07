#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class

from configuration.configuration import Configuration #class


class InputRecomDefinition:

    COS_CB_MEAN:str = "cosCBmean"
    COS_CB_WINDOW3:str = "cosCBwindow3"
    THE_MOST_POPULAR:str = "theMostPopular"
    W2V_POSNEG_MEAN:str = "w2vPosnegMean"
    W2V_POSNEG_WINDOW3:str = "w2vPosnegWindow3"


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


    @staticmethod
    def exportPairOfRecomIdsAndRecomDescrs(datasetID:str):

        recom:str = "Recom"

        rIDsCB:List[str] = [recom + InputRecomDefinition.COS_CB_MEAN.title(), recom + InputRecomDefinition.COS_CB_WINDOW3.title()]
        rDescsCB:List[RecommenderDescription] = [InputRecomDefinition.exportRDescCBmean(datasetID), InputRecomDefinition.exportRDescCBwindow3(datasetID)]

        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPositiveWindow10", "RecomW2vPosnegMean", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPositiveMax, rDescW2vPositiveWindow10, rDescW2vPosnegMean, rDescW2vPosnegWindow10]
        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPosnegMax", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPosnegMax, rDescW2vPosnegMax, rDescW2vPosnegWindow10]

        rIDsW2V:List[str] = [recom + InputRecomDefinition.W2V_POSNEG_MEAN.title(), recom + InputRecomDefinition.W2V_POSNEG_WINDOW3.title()]
        rDescsW2V:List[RecommenderDescription] = [InputRecomDefinition.exportRDescW2vPosnegMean(datasetID), InputRecomDefinition.exportRDescW2vPosnegWindow3(datasetID)]

        rIDs:List[str] = [recom + InputRecomDefinition.THE_MOST_POPULAR.title()] + rIDsCB + rIDsW2V
        rDescs:List[RecommenderDescription] = [InputRecomDefinition.exportRDescTheMostPopular(datasetID)] + rDescsCB + rDescsW2V

        return (rIDs, rDescs)
