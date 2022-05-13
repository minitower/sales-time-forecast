import pandas as pd
from fileWork import *

class Model:

    def __init__(self, df):
        """
        Class for build model to forecast data and
        other helpfull statistic methods
        """
        self.df = df
        self.fw = FileWork("main.log")


    def statistics(self, printStat=True, writeStat=True):
        """
        Func for calculate some statistics for module and 
        print in console and/or write in main.log file in 
        project root

        Args:
            print (bool, default: True) - if print - print main statistic on console/terminal
            write (bool, default: True) - if write - write main statistic on ${PROJECT_PATH}/main.log
        """
        successSales = self.df.loc[self.df['isSuccsess'] == 1]
        lostSales = self.df.loc[self.df['isSuccsess'] == 0]
        maleSales = self.df.loc[self.df['gender'] == 'm']
        femaleSales = self.df.loc[self.df['gender'] == 'f']
        knownAge = self.df.loc[self.df['gender'] == "None"]

        if printStat:
            print("MAIN STATISTICS OF SALES IN DB")
            print("__________________________________________________________________________")
            print("Percent of success sales: ", len(successSales) / len(self.df))
            print("Percent of lost sales: ", len(lostSales) / len(self.df))
            print("Percent of known sales: ", (len(successSales) + len(lostSales)) / len(self.df))
            print("Percent of female customers: ", len(femaleSales) / len(self.df))
            print("Percent of male customers: ", len(maleSales) / len(self.df))
            print("Percent of gender of customers in DB: ", (len(maleSales) + len(femaleSales)) / len(self.df))
            print(f"Percent of known age of customers in DB: {len(knownAge) / len(self.df)}")
            print(f"Percent of unknown age of customers in DB: {1 - (len(knownAge) / len(self.df))}")

        if writeStat:
            self.fw.writeInFile(message="MAIN STATISTICS OF SALES IN DB")
            self.fw.writeInFile(message="__________________________________________________________________________")
            self.fw.writeInFile(message=f"Percent of success sales: {len(successSales) / len(self.df)}")
            self.fw.writeInFile(message=f"Percent of lost sales: {len(lostSales) / len(self.df)}")
            self.fw.writeInFile(message=f"Percent of known sales: {(len(successSales) + len(lostSales)) / len(self.df)}")
            self.fw.writeInFile(message=f"Percent of female customers: {len(femaleSales) / len(self.df)}")
            self.fw.writeInFile(message=f"Percent of male customers: {len(maleSales) / len(self.df)}")
            self.fw.writeInFile(message=f"Percent of gender of customers in DB: {(len(maleSales) + len(femaleSales)) / len(self.df)}")
            self.fw.writeInFile(message=f"Percent of unknown age of customers in DB: {1 - (len(knownAge) / len(self.df))}")
            self.fw.writeInFile(message=f"Percent of known age of customers in DB: {len(knownAge) / len(self.df)}")