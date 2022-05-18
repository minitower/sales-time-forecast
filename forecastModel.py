from unittest import result
import pandas as pd
from sklearn import preprocessing
from fileWork import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import numpy as np


class Model:

    def __init__(self, df, mainLogFilename = "./results/log/main.log", 
                    logitLogFilename="./results/log/logit.log",
                    logitModelFilename = "./results/models/logit.pickle",
                    fwLassoModel = "./results/models/lasso.pickle",
                    fwSGDModel = "./results/models/sgd.pickle",
                    lassoLogFilename = "./results/log/lasso.log",
                    sgdLogFilename = "./results/log/sgd.log",
                    pathesDict=None):
        """
        Class for build model to forecast data and
        other helpfull statistic methods

        Args:
            mainLogFilenaoc[df['var']>900*i, 'interval_var'] = 15*i
string, default: "./results/log/logit.log"):
                 Env(dict, default: None): most easy way to update pathes
                for log file. Just build .env file (like in .env_example)
                and put this env on dict with param name. 
        """
        self.df = df
        self.logitModelFilename = logitModelFilename
        if pathesDict is None:
            self.fwMain = FileWork(mainLogFilename)
            self.fwLogit = FileWork(logitLogFilename)
            self.fwLogitModel = FileWork(logitModelFilename)
            self.fwLassoModel = FileWork(fwLassoModel)
            self.fwSGDModel = FileWork(fwSGDModel)
            self.fwlassoLogFilename = FileWork(lassoLogFilename)
            self.fwsgdLogFilename = FileWork(sgdLogFilename)
        else:
            self.fwMain = FileWork(pathesDict['mainLogFilename'])
            self.fwLogit = FileWork(pathesDict['logitLogFilename'])
            self.fwLogitModel = FileWork(pathesDict['logitModelFilename'])
            self.fwLassoModel = FileWork(pathesDict['fwLassoModel'])
            self.fwSGDModel = FileWork(pathesDict['fwSGDModel'])
            self.fwlassoLogFilename = FileWork(pathesDict['lassoLogFilename'])
            self.fwsgdLogFilename = FileWork(pathesDict['sgdLogFilename'])

    def statistics(self, printStat=True, writeStat=True):
        """
        Func for calculate some statistics for module and 
        print in console and/or write in main.log file in 
        project root

        Args:
            print (bool, default: True) - if print - print main statistic on console/terminal
            write (bool, default: True) - if write - write main statistic on ${PROJECT_PATH}/main.log
        """
        successSales = self.df.loc[self.df['isSuccess'] == 1]
        lostSales = self.df.loc[self.df['isSuccess'] == 0]
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

    
    def preprocessor(self, columns=['isSuccess', 'numOfCalls'], 
                    depVar = 'isSuccess', withNoneValue=False, 
                    saveDataCut=0.5, test_size=0.25, returnData=False, 
                    intervalVar=True):
        """
        Func for preprocess pd.DataFrame with data from DB to 
        analysis.
        
        Args:
            columns (list, default var state on source code): number of
                columns in final Data Frame
            depVar(string, default isSuccess): state a dependent variable
                for next modeling
            withAgeOnly(bool, default: True): if True - use rows with known
                age only.
            saveDataCut(float, default: 0.5): parameter for save fix part of data
            test_size(float, default: 0.25): parameter for set size of the test 
                Data Frame (in this module Data Frame only have test and train
                dataset)
            returnData(bool, default: False): if True - return four (4) dataset 
                after split (x, y - test and train). Most usefull for debug
            intervalVar(bool, default: True): if True - calculate interval var 
                from var and add it in final data frame
        """
        df = self.df[columns]
        
        if intervalVar:
            df = df.loc[df['fake_approve'] == 0]
            df['interval_var'] = [np.NaN]*len(df)

            df.loc[df['var']<900, 'interval_var'] = 0
            for i in range(0,30):
                df.loc[df['var']>900*i, 'interval_var'] = 15*i
            df.loc[df['var']>26100, 'interval_var'] = np.NaN
        
        if 'click_id' in columns:
            df.set_index('click_id', drop=True)
            
        print(len(df))
        if not withNoneValue and 'age' in df.columns.values:
            df = df.loc[(df['age'].isna() == False) & (df['gender'] != "None")]
            if len(df) / len(self.df) <= saveDataCut:
                raise ValueError("saveDataCut value is higer then part of dataset, "
                                    "which will be delete from original Data Frame")
        dictGender = {'m': 0, 'f':1}
        if 'gender' in df.columns.values:
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

        self.logitAccuracy = testScore
        
        if printStat:
            print("LOGIT MODEL STATS")
            print("__________________________________________________________________________")
            print(f"Mean accuracy on test dataset: {testScore}")
            print(f"Mean accuracy on full dataset: {fullScore}")
            print(f"Diff in test data set: {fullScore - testScore}")
            print(f"Test predict raiser then math expectation on {0.5 - testScore}")
            print(f"Full predict raiser then math expectation on {0.5 - fullScore}")

        
        if writeStat:
            self.fwLogit.writeInFile(message="LOGIT MODEL STATS")
            self.fwLogit.writeInFile(message="__________________________________________________________________________")
            self.fwLogit.writeInFile(message=f"Mean accuracy on test dataset: {testScore}")
            self.fwLogit.writeInFile(message=f"Mean accuracy on full dataset: {fullScore}")
            self.fwLogit.writeInFile(message=f"Diff in test data set: {fullScore - testScore}")
            self.fwLogit.writeInFile(message=f"Test predict raiser then math expectation on {0.5 - testScore}")
            self.fwLogit.writeInFile(message=f"Full predict raiser then math expectation on {0.5 - fullScore}")
        if saveModel:
            try:
                pickle.dump(clf, self.logitModelFilename)
            except TypeError:
                check = self.fwLogitModel.createFile()
                if not check:
                    raise FileNotFoundError(f"Can't create/open file {self.logitModelFilename}")
            if printStat:
                print(f"Logit model save on {self.logitModelFilename}!")
            if writeStat:
                self.fwLogit.writeInFile(message=f"Logit model save on {self.logitModelFilename}!")
        
        
    def lassoModel(self, printStat=True, writeStat=True, saveModel=True):
        """
        Func for build and test Lasso model

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
            clf = Lasso(alpha=0.1).fit(self.X_train, self.y_train)
        except ValueError:
            raise ValueError("Find None value of dataset! Please, use"
                                " age only (in preprocessing)")
        testScore = clf.score(self.X_test, self.y_test)
        fullScore = clf.score(self.full_X, self.full_y)

        self.lassoAccuracy = testScore
        
        if printStat:
            print("LASSO MODEL STATS")
            print("__________________________________________________________________________")
            print(f"Mean accuracy on test dataset: {testScore}")
            print(f"Mean accuracy on full dataset: {fullScore}")
            print(f"Diff in test data set: {fullScore - testScore}")
            print(f"Test predict raiser then math expectation on {0.5 - testScore}")
            print(f"Full predict raiser then math expectation on {0.5 - fullScore}")

        
        if writeStat:
            self.fwLogit.writeInFile(message="LASSO MODEL STATS")
            self.fwLogit.writeInFile(message="__________________________________________________________________________")
            self.fwLogit.writeInFile(message=f"Mean accuracy on test dataset: {testScore}")
            self.fwLogit.writeInFile(message=f"Mean accuracy on full dataset: {fullScore}")
            self.fwLogit.writeInFile(message=f"Diff in test data set: {fullScore - testScore}")
            self.fwLogit.writeInFile(message=f"Test predict raiser then math expectation on {0.5 -testScore}")
            self.fwLogit.writeInFile(message=f"Full predict raiser then math expectation on {0.5 - fullScore}")
        if saveModel:
            try:
                pickle.dump(clf, self.fwLassoModel)
            except TypeError:
                check = self.fwLassoModel.createFile()
                if not check:
                    raise FileNotFoundError(f"Can't create/open file {self.logitModelFilename}")
            if printStat:
                print(f"LASSO model save on {self.fwLassoModel}!")
            if writeStat:
                self.fwLogit.writeInFile(message=f"LASSO model save on {self.fwLassoModel}!")
        
        
    def SGDModel(self, printStat=True, writeStat=True, saveModel=True):
        """
        Func for build and test Lasso model

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
            clf = SGDClassifier(loss="hinge", penalty="l2",
                                    max_iter=5).fit(self.X_train, self.y_train)
        except ValueError:
            raise ValueError("Find None value of dataset! Please, use"
                                " age only (in preprocessing)")
        testScore = clf.score(self.X_test, self.y_test)
        fullScore = clf.score(self.full_X, self.full_y)
        
        self.sgdAccuracy = testScore
        
        if printStat:
            print("SGD CLASSIFIER STATS")
            print("__________________________________________________________________________")
            print(f"Mean accuracy on test dataset: {testScore}")
            print(f"Mean accuracy on full dataset: {fullScore}")
            print(f"Diff in test data set: {fullScore - testScore}")
            print(f"Test predict raiser then math expectation on {0.5 - testScore}")
            print(f"Full predict raiser then math expectation on {0.5 - fullScore}")

        
        if writeStat:
            self.fwLogit.writeInFile(message="SGD CLASSIFIER STATS")
            self.fwLogit.writeInFile(message="__________________________________________________________________________")
            self.fwLogit.writeInFile(message=f"Mean accuracy on test dataset: {testScore}")
            self.fwLogit.writeInFile(message=f"Mean accuracy on full dataset: {fullScore}")
            self.fwLogit.writeInFile(message=f"Diff in test data set: {fullScore - testScore}")
            self.fwLogit.writeInFile(message=f"Test predict raiser then math expectation on {0.5 - testScore}")
            self.fwLogit.writeInFile(message=f"Full predict raiser then math expectation on {0.5 - fullScore}")
        if saveModel:
            try:
                pickle.dump(clf, self.fwsgdLogFilename)
            except TypeError:
                check = self.fwLogitModel.createFile()
                if not check:
                    raise FileNotFoundError(f"Can't create/open file {self.logitModelFilename}")
            if printStat:
                print(f"SGD CLASSIFIER save on {self.logitModelFilename}!")
            if writeStat:
                self.fwLogit.writeInFile(message=f"SGD CLASSIFIER save on {self.logitModelFilename}!")
    
    
    def findOptimalColumns(self, model=None):
        """
        Func for find more complex model for some columns from 
        "maxColumns" list. Model param accurate some of models
        
        Args:
            model (string, default: None): param for state one of the model
                from class. Expected value: "logit", "lasso", "sgd"
        """
        if model is None:
            raise ValueError("MODEL PARAM DIDN'T STATE!")
        
        columnsArr = [['isSuccess', 'numOfCalls'], ['isSuccess', 'var'], 
                      ['isSuccess', 'interval_var'], ['isSuccess', 'numOfCalls', 'interval_var']
                      ]
        
        if "logit":
            self.preprocessor(columns=colArr)