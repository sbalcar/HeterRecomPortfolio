#!/usr/bin/python3

import pandas as pd
import numpy as np

from pandas.core.frame import DataFrame #class
from configuration.configuration import Configuration #class
from typing import List #class
from typing import Set #class

from sklearn.metrics import *
from sklearn.preprocessing import normalize


from datasets.aDataset import ADataset #class
from datasets.datasetML import DatasetML #class
from datasets.datasetRetailrocket import DatasetRetailRocket #class
from datasets.datasetST import DatasetST #class



class ToolMMR():

    def init(self, trainDataset:ADataset):
        if  not isinstance(trainDataset, ADataset):
            raise ValueError("Argument trainDataset isn't type ADataset.")

        if type(trainDataset) is DatasetML:
            cbDataPath = Configuration.cbML1MDataFileWithPathOHE
        elif type(trainDataset) is DatasetRetailRocket:
            return None

        elif type(trainDataset) is DatasetST:
            cbDataPath = Configuration.cbSTDataFileWithPathOHE

        mmrCBFeatures = pd.read_csv(cbDataPath, sep=",", header=0, index_col=0)
        # print(mmrCBFeatures)

        dfCBSim = 1 - pairwise_distances(mmrCBFeatures, metric="cosine")
        self.mmrCBSim: DataFrame = DataFrame(data=dfCBSim, index=mmrCBFeatures.index, columns=mmrCBFeatures.index)
        # print(self.CBSim.shape)

    def _argmax(self, keys, f):
        return max(keys, key=f)

    def _objects_similarity(self, i:int, j:int):
        # print(i,j, self.mmrCBSim.at[i, j])
        try:
            return self.mmrCBSim.at[i, j]
        except:
            print("miss")
            return 0

    def mmr_sorted_with_prefix(self, lambda_:float, results, prefix, length:int):
        selected = pd.Series(dtype=np.float64)
        prefix = set(prefix)
        docs = set(results.index)
        while (len(selected) < len(docs)) and (len(selected) < length):
            remaining = docs - set(selected.index)
            mmr_score = lambda x: lambda_ * results[x] - (1 - lambda_) * max(
                [self._objects_similarity(x, y) for y in set(selected.index).union(prefix) - {x}] or [
                    0])  # TODO: self.mmr_objects_similarity
            next_selected = self._argmax(remaining, mmr_score)
            mmrVal = lambda_ + (lambda_ * results[next_selected] - (1 - lambda_) * max(
                [self._objects_similarity(next_selected, y) for y in set(selected.index).union(prefix) - {next_selected}] or [0]))
            selected = selected.append(pd.Series(mmrVal, index=[next_selected]))
            # selected[next_selected] = mmrVal

        return selected

    def mmr_sorted(self, lambda_:float, results, length:int):
        print("lambda_: " + str(lambda_))
        """Sort a list of docs by Maximal marginal relevance
        Performs maximal marginal relevance sorting on a set of
        documents as described by Carbonell and Goldstein (1998)
        in their paper "The Use of MMR, Diversity-Based Reranking
        for Reordering Documents and Producing Summaries"
        :param docs: a set of documents to be ranked
                      by maximal marginal relevance
        :param q: query to which the documents are results
        :param lambda_: lambda parameter, a float between 0 and 1
        :return: a (document, mmr score) ordered dictionary of the docs
                given in the first argument, ordered my MMR
        """
        print("enter to MMR")

        selected = pd.Series(dtype=np.float64)
        docs = set(results.index)
        while (len(selected) < len(docs)) and (len(selected) < length):
            remaining = docs - set(selected.index)
            mmr_score = lambda x: lambda_ * results[x] - (1 - lambda_) * max(
                [self._objects_similarity(x, y) for y in set(selected.index) - {x}] or [
                    0])  # TODO: self.mmr_objects_similarity
            next_selected = self._argmax(remaining, mmr_score)
            mmrVal = lambda_ + (lambda_ * results[next_selected] - (1 - lambda_) * max(
                [self._objects_similarity(next_selected, y) for y in set(selected.index) - {next_selected}] or [0]))
            selected = selected.append(pd.Series(mmrVal, index=[next_selected]))
            # selected[next_selected] = mmrVal

        return selected
