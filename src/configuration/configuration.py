#!/usr/bin/python3

import os

class Configuration:

    dataDirectory:str = ".." + os.sep + "data"

    cbML1MDataFileWithPathTFIDF:str = dataDirectory + os.sep + "cbDataTFIDF.txt"
    cbML1MDataFileWithPathOHE:str = dataDirectory + os.sep + "cbDataOHE.txt"

    cbSTDataFileWithPathTFIDF: str = dataDirectory + os.sep + "SLANtourCBFeaturesTFIDF.txt"
    cbSTDataFileWithPathOHE:str = dataDirectory + os.sep + "SLANtourCBFeaturesOHE.txt"

    cbRRDataFileWithPathOHE:str = dataDirectory + os.sep + "simMatrixRR.npz"


    modelDirectory:str =  ".." + os.sep + "models"


    imagesDirectory:str =  ".." + os.sep + "images"


    resultsDirectory:str =  ".." + os.sep + "results"