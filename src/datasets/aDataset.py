#!/usr/bin/python3

from abc import ABC, abstractmethod

class ADataset(ABC):

    @staticmethod
    def readDatasets():
        assert False, "this needs to be overridden"

