#!/usr/bin/python3

from typing import List

from recommenderDescription.recommenderDescription import RecommenderDescription #class

from recommender.recommenderTheMostPopular import RecommenderTheMostPopular #class
from recommender.recommenderCosineCB import RecommenderCosineCB #class
from recommender.recommenderW2V import RecommenderW2V #class

from datasets.ratings import Ratings #class
from datasets.users import Users #class
from datasets.items import Items #class
from datasets.behaviours import Behaviours #class

from configuration.configuration import Configuration #class

from pandas.core.frame import DataFrame #class

from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.aPenalization import APenalization #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import PenalUsingReduceRelevance #class
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import penaltyStatic #function
from aggregation.toolsDHontNF.penalizationOfResultsByNegImpFeedback.penalUsingReduceRelevance import penaltyLinear #function


class InputRecomDefinition:

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
                 RecommenderW2V.ARG_DATASET_ID: datasetID})



    @staticmethod
    def exportPairOfRecomIdsAndRecomDescrs(datasetID:str):

        rIDsCB:List[str] = ["RecomCBmean", "RecomCBwindow3"]
        rDescsCB:List[RecommenderDescription] = [InputRecomDefinition.exportRDescCBmean(datasetID), InputRecomDefinition.exportRDescCBwindow3(datasetID)]

        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPositiveWindow10", "RecomW2vPosnegMean", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPositiveMax, rDescW2vPositiveWindow10, rDescW2vPosnegMean, rDescW2vPosnegWindow10]
        #rIDsW2V:List[str] = ["RecomW2vPositiveMax", "RecomW2vPosnegMax", "RecomW2vPosnegWindow10"]
        #rDescsW2V:List[RecommenderDescription] = [rDescW2vPosnegMax, rDescW2vPosnegMax, rDescW2vPosnegWindow10]
        rIDsW2V:List[str] = ["RecomW2vPosnegMax", "RecomW2vPosnegWindow3"]
        rDescsW2V:List[RecommenderDescription] = [InputRecomDefinition.exportRDescW2vPosnegMean(datasetID), InputRecomDefinition.exportRDescW2vPosnegWindow3(datasetID)]

        rIDs:List[str] = ["RecomTheMostPopular"] + rIDsCB + rIDsW2V
        rDescs:List[RecommenderDescription] = [InputRecomDefinition.exportRDescTheMostPopular(datasetID)] + rDescsCB + rDescsW2V

        return (rIDs, rDescs)
