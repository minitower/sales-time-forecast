from statistics import mode
import pandas as pd
from fileWork import *
from forecastModel import Model
from dotenv import load_dotenv
import numpy as np

def main():
    """
    Main func of sales-time-forecast project. 
    Contain all other modules
    """
    pathesDict={"mainLogFilename": os.environ.get("MAINLOGFILENAME"),
                "logitLogFilename": os.environ.get("LOGITLOGFILENAME"),
                "logitModelFilename": os.environ.get("LOGITMODELFILENAME"),
                "fwLassoModel": os.environ.get("FWLASSOMODEL"),
                "fwSGDModel": os.environ.get("FWSGDMODEL"),
                "lassoLogFilename": os.environ.get("LASSOLOGFILENAME"),
                "sgdLogFilename": os.environ.get("SGDLOGFILENAME")}
    for i in pathesDict.values():
        if os.path.exists(i):
            FileWork.removeFiles(filepath = i)

    df = pd.read_csv("./concatResponse.csv")
    
    model = Model(df=df, pathesDict=pathesDict)
    logitColumns = model.findOptimalColumns(model="logit")
    print(f"Logit max accuracy: {model.maxAccuracy}")
    logitAccuracy = model.maxAccuracy
    lassoColumns = model.findOptimalColumns(model="lasso")
    print(f"Lasso max accuracy: {model.maxAccuracy}")
    lassoAccuracy = model.maxAccuracy
    sgdColumns = model.findOptimalColumns(model="sgd")
    print(f"SGD max accuracy: {model.maxAccuracy}")
    sgdAccuracy = model.maxAccuracy
    if logitAccuracy >= lassoAccuracy and \
        logitAccuracy >= sgdAccuracy:
            print(f"True columns: {logitColumns}")
    elif lassoAccuracy>=logitAccuracy and \
        lassoAccuracy>=sgdAccuracy:
            print(f"True columns: {lassoColumns}")
    elif sgdAccuracy>=logitAccuracy and \
        sgdAccuracy>=lassoAccuracy:
            print(f"True columns: {sgdColumns}")
    

if __name__ == "__main__":
    load_dotenv()
    main()