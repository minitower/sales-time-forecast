import pandas as pd
from fileWork import *
from forecastModel import Model
from dotenv import load_dotenv

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
    model.preprocessor()
    model.logitModel()
    model.SGDModel()
    model.lassoModel()

if __name__ == "__main__":
    load_dotenv()
    main()