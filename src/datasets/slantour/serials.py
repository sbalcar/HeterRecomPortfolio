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
        serialFile: str = ".." + os.sep + "datasets" + os.sep + "slantour" + os.sep + "serial_only_data_from_2018.csv"
        serialsDF: pd.DataFrame = pd.read_csv(serialFile, sep=';', header=0)
        serialsDF.rename(columns={"AVG(prumerna_cena)": "prumerna_cena", "AVG(prumerna_cena_noc)": "prumerna_cena_noc", "AVG(delka)": "delka"}, inplace=True)
        serialsDF.set_index("id_serial", drop=False, inplace=True)
        serialFileMonths: str = ".." + os.sep + "datasets" + os.sep + "slantour" + os.sep + "serial_months_from_2018.csv"
        serialsDFMonths: pd.DataFrame = pd.read_csv(serialFileMonths, sep=';', header=0, index_col = 0)
        
        #print(serialsDFMonths.shape)
        
        for i in range(1,13):
            serialsDF.insert(0, "month_"+str(i),0)
        
        for index, row in serialsDFMonths.iterrows():
            i = int(row["month_from"])
            j = int(row["month_to"])
            while True:   
                try:      
                    serialsDF.loc[index, "month_"+str(i)] = 1
                except:
                    pass
            
                if (i == j) or (j <= 0) or (j > 12):
                    break
                else:            
                    i = (i%12) + 1
        
        #print(serialsDF.describe())
        
        return serialsDF
