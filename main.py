import pandas as pd
from fileWork import *
from forecastModel import Model

def main():
    """
    Main func of sales-time-forecast project. 
    Contain all other modules
    """
    df = pd.read_csv("./concatResponse.csv")
    model = Model(df=df)
    model.statistics(printStat=True, writeStat=True)

if __name__ == "__main__":
    main()