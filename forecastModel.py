import pandas as pd
from fileWork import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle


class Model:

    def __init__(self, df, mainLogFilename = "./results/log/main.log", 
                    logFilename="./results/log/logit.log",
                    logitModelFilename = "./results/models/logit.pickle"):
        """
        Class for build model to forecast data and
        other helpfull statistic methods
        """
        self.df = df
        self.logitModelFilename = logitModelFilename
        self.fwMain = FileWork(mainLogFilename)
        self.fwLogit = FileWork(logFilename)
        self.fwLogitModel = FileWork()


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
            self.fwMain.writeInFile(message="MAIN STATISTICS OF SALES IN DB")
            self.fwMain.writeInFile(message="__________________________________________________________________________")
            self.fwMain.writeInFile(message=f"Percent of success sales: {len(successSales) / len(self.df)}")
            self.fwMain.writeInFile(message=f"Percent of lost sales: {len(lostSales) / len(self.df)}")
            self.fwMain.writeInFile(message=f"Percent of known sales: {(len(successSales) + len(lostSales)) / len(self.df)}")
            self.fwMain.writeInFile(message=f"Percent of female customers: {len(femaleSales) / len(self.df)}")
            self.fwMain.writeInFile(message=f"Percent of male customers: {len(maleSales) / len(self.df)}")
            self.fwMain.writeInFile(message=f"Percent of gender of customers in DB: {(len(maleSales) + len(femaleSales)) / len(self.df)}")
            self.fwMain.writeInFile(message=f"Percent of unknown age of customers in DB: {1 - (len(knownAge) / len(self.df))}")
            self.fwMain.writeInFile(message=f"Percent of known age of customers in DB: {len(knownAge) / len(self.df)}")

    
    def preprocessor(self, columns=['click_id', 'age', 'gender', 
                    'var', 'isSuccsess', 'numOfCalls'], 
                    depVar = 'isSuccsess', withNoneValue=False, 
                    saveDataCut=0.5, test_size=0.25, returnData=False):
        """
        Func for preprocess pd.DataFrame with data from DB to 
        analysis.
        
        Args:
            columns (list, default var state on source code): number of
                columns in final Data Frame
            depVar(string, default isSuccsess): state a dependent variable
                for next modeling
            withAgeOnly(bool, default: True): if True - use rows with known
                age only.
            saveDataCut(float, default: 0.5): parameter for save fix part of data
            test_size(float, default: 0.25): parameter for set size of the test 
                Data Frame (in this module Data Frame only have test and train
                dataset)
            returnData(bool, default: False): if True - return four (4) dataset 
                after split (x, y - test and train). Most usefull for debug
        """
        df = self.df[columns]
        if 'click_id' in columns:
            df.set_index('click_id', drop=True)
        print(len(df))
        if not withNoneValue:
            df = df.loc[(df['age'].isna() == False) & (df['gender'] != "None")]
            if len(df) / len(self.df) <= saveDataCut:
                raise ValueError("saveDataCut value is higer then part of dataset, "
                                    "which will be delete from original Data Frame")
        dictGender = {'m': 0, 'f':1}
        df['gender'] = df['gender'].map(dictGender)
        columns.remove(depVar)
        x = df[columns].values
        y = df[depVar].values
        self.X_train, self.X_test, \
                self.y_train, self.y_test = train_test_split(x, y, test_size=test_size)
        self.full_X, self.full_y = x, y
        if returnData:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def logitModel(self, printStat=True, writeStat=True, saveModel=True):
        """
        Func for build and test logit model

        Args:
            printStat(bool, default: True): if True - print info
                about model in console/terminal
            writeStat(bool, default:True): if True - write info 
                about model in results folder
            logFilename(string, default:'./results/log/logit.log'):
                path to logit model log file
            saveModel(bool, default: True): save model on ../results/models/logit.pickle
            modelFilename(string, default: "./results/models/logit.pickle"):
                path to logit model pickle file (for addictive analysis)
        """
        try:
            clf = LogisticRegression().fit(self.X_train, self.y_train)
        except ValueError:
            raise ValueError("Find None value of dataset! Please, use"
                                " age only (in preprocessing)")
        testScore = clf.score(self.X_test, self.y_test)
        fullScore = clf.score(self.full_X, self.full_y)

        if printStat:
            print("LOGIT MODEL STATS")
            print("__________________________________________________________________________")
            print(f"Mean accuracy on test dataset: {testScore}")
            print(f"Mean accuracy on full dataset: {fullScore}")
        
        if writeStat:
            self.fwLogit.writeInFile(message="LOGIT MODEL STATS")
            self.fwLogit.writeInFile(message="__________________________________________________________________________")
            self.fwLogit.writeInFile(message=f"Mean accuracy on test dataset: {testScore}")
            self.fwLogit.writeInFile(message=f"Mean accuracy on full dataset: {fullScore}")
        
        if saveModel:
            try:
                pickle.dump(clf, modelFilename)
            except TypeError:
                
            if printStat:
                print(f"Logit model save on {modelFilename}!")
            if writeStat:
                self.fwLogit.writeInFile(message=f"Logit model save on {modelFilename}!")
        
        