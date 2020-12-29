#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from datasets.aDataset import ADataset #class

from datasets.slantour.events import Events #class
from datasets.slantour.serials import Serials #class


class DatasetST(ADataset):

    def __init__(self, datasetID:str, eventsDF:DataFrame, serialsDF:DataFrame):
        if type(datasetID) is not str:
            raise ValueError("Argument datasetID isn't type str.")
        if type(eventsDF) is not DataFrame:
            raise ValueError("Argument eventsDF isn't type DataFrame.")
        if type(serialsDF) is not DataFrame:
            raise ValueError("Argument serialsDF isn't type DataFrame.")

        self.datasetID = datasetID
        self.eventsDF:DataFrame = eventsDF
        self.serialsDF:DataFrame = serialsDF

    @staticmethod
    def readDatasets():
        # dataset reading
        eventsDF:DataFrame = Events.readFromFile()
        serialsDF:DataFrame = Serials.readFromFile()

        return DatasetST("stDivAll", eventsDF, serialsDF)


    def getTheMostSold(self):

        eventsTrainDF:DataFrame = self.eventsDF

        # removing catalog data
        eventsTrainDF = eventsTrainDF.loc[eventsTrainDF[Events.COL_OBJECT_ID] != 0]

        # ratingsSum:Dataframe<(movieId:int, ratings:int)>
        eventsTransSumDF:DataFrame = DataFrame(eventsTrainDF.groupby(Events.COL_OBJECT_ID)[Events.COL_USER_ID].count())

        # sortedAsceventsTransCountDF:Dataframe<(movieId:int, ratings:int)>
        sortedAsceventsTransCountDF:DataFrame = eventsTransSumDF.sort_values(by=Events.COL_USER_ID, ascending=False)
        #print(sortedAsceventsTransCountDF)

        return sortedAsceventsTransCountDF
