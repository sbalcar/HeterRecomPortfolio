#!/usr/bin/python3

from pandas.core.frame import DataFrame #class

from datasets.aDataset import ADataset #class

from datasets.retailrocket.events import Events #class
from datasets.retailrocket.categoryTree import CategoryTree #class
from datasets.retailrocket.itemProperties import ItemProperties #class

class DatasetRetailRocket(ADataset):

    def __init__(self, datasetID:str, eventsDF:DataFrame, categoryTreeDF:DataFrame, itemPropertiesDF:DataFrame):
        if type(datasetID) is not str:
            raise ValueError("Argument datasetID isn't type str.")
        if type(eventsDF) is not DataFrame:
            raise ValueError("Argument eventsDF isn't type DataFrame.")
        if type(categoryTreeDF) is not DataFrame:
            raise ValueError("Argument categoryTreeDF isn't type DataFrame.")
        if type(itemPropertiesDF) is not DataFrame:
            raise ValueError("Argument itemPropertiesDF isn't type DataFrame.")

        self.datasetID = datasetID
        self.eventsDF:DataFrame = eventsDF
        self.categoryTreeDF:DataFrame = categoryTreeDF
        self.itemPropertiesDF:DataFrame = itemPropertiesDF

    @staticmethod
    def readDatasets():

        eventsDF:DataFrame = Events.readFromFile()
        categoryTreeDF:DataFrame = CategoryTree.readFromFile()
        itemPropertiesDF:DataFrame = ItemProperties.readFromFile()

        return DatasetRetailRocket("rrDivAll", eventsDF, categoryTreeDF, itemPropertiesDF)



    def getTheMostSold(self):

        eventsTrainDF:DataFrame = self.eventsDF

        # ratingsSum:Dataframe<(timestamp:int, visitorid:int, event:str, itemid:int, transactionid:int)>
        eventsTransDF:DataFrame = eventsTrainDF.loc[eventsTrainDF[Events.COL_EVENT] == "transaction"]

        # ratingsSum:Dataframe<(movieId:int, ratings:int)>
        eventsTransSumDF:DataFrame = DataFrame(eventsTransDF.groupby(Events.COL_ITEM_ID)[Events.COL_EVENT].count())

        # sortedAsceventsTransCountDF:Dataframe<(movieId:int, ratings:int)>
        sortedAsceventsTransCountDF:DataFrame = eventsTransSumDF.sort_values(by=Events.COL_EVENT, ascending=False)
        #print(sortedAsceventsTransCountDF)

        return sortedAsceventsTransCountDF
