#!/usr/bin/python3

from history.aHistory import AHistory #class


class APortfolioDescription:

    def __init__(self):
        raise Exception("APortfolioDescription is abstract class, can't be instanced")

    def getPortfolioID(self):
        pass

    def exportPortfolio(self, batchID:str, jobID:str, history:AHistory):
        pass