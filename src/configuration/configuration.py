#!/usr/bin/python3

import os

class Configuration:

    dataDirectory:str = ".." + os.sep + "data"

    cbDataFileWithPathTFIDF:str = dataDirectory + os.sep + "cbDataTFIDF.txt"
    cbDataFileWithPathOHE:str = dataDirectory + os.sep + "cbDataOHE.txt"



    modelDirectory:str =  ".." + os.sep + "models"


    imagesDirectory:str =  ".." + os.sep + "images"


    resultsDirectory:str =  ".." + os.sep + "results"