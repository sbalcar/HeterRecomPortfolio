#!/usr/bin/python3


class APortfolioDescription:

    def __init__(self):
        raise Exception("APortfolioDescription is abstract class, can't be instanced")

    def getPortfolioID(self):
        pass

    def exportPortfolio(self):
        pass