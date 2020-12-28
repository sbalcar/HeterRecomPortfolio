#!/usr/bin/python3

import io
import csv
from typing import List

from pandas.core.frame import DataFrame #class

import pandas as pd

import os

class Serials:

    ID_RECORD:str = "id_record"
    ID_SERIAL:str = "id_serial"
    ID_AZJEZD:str = "id_zajezd"
    OD:str = "od"
    DO:str = "do"
    NAZEV:str = "nazev"
    POPISEK:str = "popisek"
    ID_TYP:str = "id_typ"
    STRAVA:str = "strava"
    DOPRAVA:str = "doprava"
    UBYTOVANI:str = "ubytovani"
    UBYTOVANI_KATEGORIE:str = "ubytovani_kategorie"
    ZEME:str = "zeme"
    DESTINACE:str = "destinace"
    PRUMERNA_CENA:str = "prumerna_cena"
    PRUMERNA_CENA_NOC:str = "prumerna_cena_noc"
    MIN_CENA:str = "min_cena"
    MIN_CENA_NOC:str = "min_cena_noc"
    SLEVA:str = "sleva"
    DELKA:str = "delka"
    INFORMACE_LIST:str = "informace_list"
    VALID_FROM:str = "valid_from"
    VALID_TO:str = "valid_to"


    @staticmethod
    def readFromFile():
        serialFile: str = ".." + os.sep + "datasets" + os.sep + "slantour" + os.sep + "new_serial_table.csv"

        serialsDF: DataFrame = pd.read_csv(serialFile, sep=',', usecols=range(23), header=0, encoding="ISO-8859-1", engine='python')
        #serialsDF.columns = [Items.COL_MOVIEID, Items.COL_MOVIETITLE, Items.COL_GENRES]

        return serialsDF
