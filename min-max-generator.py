#https://aegis4048.github.io/transforming-non-normal-distribution-to-normal-distribution

from dhis2 import Api, pretty_json
import json
import pandas as pd
import numpy as np
import math
from scipy.stats import median_abs_deviation
from scipy import stats
from scipy.special import inv_boxcox
from tqdm import tqdm
import warnings
import argparse


## INITIALISATION
warnings.filterwarnings("ignore", category=RuntimeWarning) #scipy spamming with runtime values
parser=argparse.ArgumentParser()

parser.add_argument("--conf", help="Path to configuration file")
parser.add_argument("--auth", help="Path to authentication file")
parser.add_argument("--dryrun", help="Do not post min-max values to server", action='store_true')
parser.add_argument("--file", help="Save results to file. --file ALL for all values, --file OUTLIERS to only include outliers")

def initialise(arguments):
    global conf, api, result, endDate, startDate, fileOutput, fileResult, dryRun

    # Configuration
    if arguments.conf == None:
        print("Conf file must be specified with --conf argument")
        raise Exception("Parsing config file")
    else: 
        # Load conf file
        with open(arguments.conf) as file:
            conf = json.load(file)

    # API
    if arguments.auth == None:
        print("Auth file must be specified with --auth argument")
        raise Exception("Parsing auth file")
    else:
        # Initialise API
        api = Api.from_auth_file(arguments.auth)

    # Dry run
    if arguments.dryrun:
        dryRun = True
    else: 
        dryRun = False

    # File output
    if arguments.file != None:
        if arguments.file == "ALL":
            fileOutput = "ALL"
        elif arguments.file == "OUTLIERS":
            fileOutput = "OUTLIERS"
        else: 
            print("Unknown --file argument: ", arguments.file)
            fileOutput = False
    else:
        fileOutput = False
    fileResult = None

    # Method - sort groups in conf file, make sure we categorise correctly after size
    conf["groups"] = sorted(conf["groups"], key=lambda d: d['limitMedian']) 

    # Period - TODO - change to be dependent on period type of dataset, and make sure we don't include current period
    endDate = pd.to_datetime('today') + pd.offsets.MonthEnd() 
    startDate = endDate - pd.DateOffset(years=conf["years"])

    # Summary result
    result = {
        "missing": 0,
        "valid": 0,
        "outliers": 0,
        "errors": 0
    }

try:
    initialise(parser.parse_args())
except:
    print("Fail to initialise script, check parameters.")
    exit(1)



## Get the number of periods we are looking at in total (i.e. denominator for looking at completeness)
def getPeriodCount(ds):
    dsPeriodtype = api.get("dataSets/" + ds, params={"fields": "id,periodType"})
    periodType = dsPeriodtype.json()["periodType"]

    #subtracting one, since we don't expect any report in current period
    if periodType == 'Monthly':
        return (pd.Period(endDate, 'm') - pd.Period(startDate, 'm')).n - 1
    elif periodType == 'Quarterly':
        return (pd.Period(endDate, 'q') - pd.Period(startDate, 'q')).n - 1
    else: 
        print("Periodtype not supported: " + periodType)
        exit


## Get the reported data values
def getDataValues(ds, ou, sDate, eDate):
    dvs = api.get("dataValueSets", params={
        "dataSet": ds,
        "startDate": sDate.strftime("%Y-%m-%d"),
        "endDate": eDate.strftime("%Y-%m-%d"),
        "orgUnit": ou,
        "children": "true"
    })

    return dvs.json()["dataValues"]


## Get data element value types, so that we can ignore non-numeric ones
def getDataElementType(ds):
    dsDataElements = api.get("dataSets/" + ds, params={"fields": "id,dataSetElements[dataElement[id,valueType]]"})
    return dsDataElements.json()["dataSetElements"]


## Get the current min/max values that are NOT generated - don't want to overwrite them
def getManualMinMax(des, ous):
    mm = api.get("minMaxDataElements", params={
        "filter": "dataElement.id:in:[" + ",".join(des) + "]",
        "filter": "source.id:in:[" + ",".join(ous) + "]",
        "filter": "generated:eq:false"
    })
    return mm.json()["minMaxDataElements"]


## Analyse orgunit - data element - categoryOptionCombo combination for min/max
def findMinMax(ou, de, coc, values):
    values = pd.DataFrame(values)
            
    # if number of periods with data is less than threshold, ignore
    if values["value"].count() <= math.ceil(periodCount*conf["completenessThreshold"]):
        result["missing"] += 1
        
        #TODO: delete previously generated min/max, since this is unlikely to be accurate?
        return False
    else:
        result["valid"] += 1

    #find the right group (by size of median reported number)
    median = values["value"].median()
    for group in conf["groups"]:
        if median < group["limitMedian"]:
            break

    ##Ignore chosen method if variance is 0 or very small
    if noVariance(values):
        val_max, val_min = mmPrevMax(values, 1.5)
    elif group["method"] == "PREV_MAX" or (values["value"].min() == values["value"].max()):
        val_max, val_min = mmPrevMax(values, group["threshold"])
    elif group["method"] == "ZSCORE":
        val_max, val_min = mmZScore(values, group["threshold"])
    elif group["method"] == "MAD":
        val_max, val_min = mmMAD(values, group["threshold"])
    elif group["method"] == "BOXCOX":
        val_max, val_min = mmBoxCox(values, group["threshold"])
    else: 
        print("Unknown method")
        return False
    
    comment = group["method"]

    # Check that method worked - otherwise count as error and fall back to PrexMax with 1.5 threshold
    if math.isfinite(val_min) == False or math.isfinite(val_max) == False:
        result["errors"] += 1
        val_max, val_min = mmPrevMax(values, 1.5)
        comment += " - Error"

    # Round up/down
    val_max = math.ceil(val_max)
    val_min = math.floor(val_min)

    # Count outliers
    isOutlier = False
    if max(values["value"]) > val_max or min(values["value"]) < val_min:
        result["outliers"] += 1
        comment += " - Outlier"
        isOutlier = True
    
    minMaxObject = {
        "min": int(val_min),
        "max": int(val_max),
        "generated": True,
        "dataElement": {
            "id": de
        },
        "source": {
            "id": ou
        },
        "optionCombo": {
            "id": coc
        }
    }

    # Push the values - need to do this one by one due to API limitations
    if dryRun == False:
        response = api.post("minMaxDataElements", minMaxObject)
        if response.status_code != 201:
            print("Update failed:")
            pretty_json(minMaxObject)

    return {"val_max": val_max, "val_min": val_min, "comment": comment, "outlier": isOutlier}

# Check that the series is not constant and/or with very low variance, in which case we fall back to prev min/max
def noVariance(values):
    variance = values["value"].var()
    median = values["value"].median()

    # Check if variance is < 2% of the median
    return (100*variance)/median < 2


def mmPrevMax(values, threshold):
    val_max = max([max(values["value"]) * threshold, 10])      # never less than 10
    val_min = max([max(values["value"]) * (1 - threshold), 0]) # never below 0

    return val_max, val_min


def mmZScore(values, threshold): 
    mean = values["value"].mean()
    zscore = np.std(values)

    val_max = mean + threshold * zscore
    val_min = max([float(mean - zscore * threshold), 0]) # avoid negative
    
    return val_max, val_min


def mmMAD(values, threshold): 
    mad = values[["value"]].apply(median_abs_deviation)[0]
    median = values["value"].median()
    val_max = median + 3*mad
    val_min = max([float(median - mad * threshold), 0]) # avoid negative
    
    return val_max, val_min


def mmBoxCox(values, threshold):
    try: 
        values_transformed, lmbda = stats.boxcox(values["value"])
    except:
        return np.nan, np.nan
    mean_trans = np.mean(values_transformed)
    std_trans  = np.std(values_transformed)
    
    upper_limit_trans = mean_trans + threshold * std_trans
    lower_limit_trans = mean_trans - threshold * std_trans

    val_max = inv_boxcox(upper_limit_trans, lmbda)
    val_min = inv_boxcox(lower_limit_trans, lmbda)

    # We're not too concerned with min, so set to 0 if it can't be calculated
    if math.isfinite(val_min) == False:
        val_min = 0

    # Do some sanity checks - the methods fails for somer series
    if val_max == val_min: #min and max are the same
        return np.nan, np.nan
    elif values[values["value"] > val_max].size > (values.size/2): #more than half of the values are high outliers
        return np.nan, np.nan
    elif values[values["value"] < val_min].size > (values.size/2): #more than half of the values are low outliers
        return np.nan, np.nan
    else:
        return val_max, val_min


def filterNumeric(dvs): 
    dataElements = pd.json_normalize(getDataElementType(conf["dataset"]))
    dataElements = dataElements.rename(columns={"dataElement.id": "dataElement", "dataElement.valueType": "valueType"})
    dvs = pd.merge(dvs, dataElements, how="inner", on=["dataElement"])

    validTypes =  ["INTEGER", "INTEGER_POSITIVE", "INTEGER_ZERO_OR_POSITIVE", "NUMBER"]
    return dvs[dvs['valueType'].isin(validTypes)]


def filterManualValues(dvs): 
    manualMinMax = pd.json_normalize(getManualMinMax(dvs["dataElement"].unique().tolist(), dvs["orgUnit"].unique().tolist()))
    if manualMinMax.size > 0: 
        manualMinMax["minMaxEntry"] = manualMinMax["source.id"] + "." + manualMinMax["dataElement.id"] + "." + manualMinMax["optionCombo.id"]
        dvs = dvs[~dvs['minMaxEntry'].isin(manualMinMax["minMaxEntry"].tolist())]
    return dvs

def generateValues(ou, ds):

    # Get the number of potential periods we should expect
    global periodCount 
    periodCount = getPeriodCount(ds)

    # Get data values for the orgunit and dataset in question, and save as DataFrame
    dataValues = pd.DataFrame(getDataValues(ds, ou, startDate, endDate))

    # Add column identifying the "unique" combinations of orgunit, data element and CoC. Each of these should get a min/max
    dataValues["minMaxEntry"] = dataValues["orgUnit"] + "." + dataValues["dataElement"] + "." + dataValues["categoryOptionCombo"]

    # Exclude non-numeric data elements
    dataValues = filterNumeric(dataValues)

    # Exclude min-max that have been manually set
    dataValues = filterManualValues(dataValues)

    # Change value type to numeric
    dataValues['value'] = dataValues['value'].astype(float)

    ## For testing, build "pivoted" spreadsheet where min, max etc are included (since there is no min/max analysis anymore)
    dvPivoted =  dataValues.pivot_table(values=["value"], index=["minMaxEntry"], columns=["period"], aggfunc=np.sum)

    # Iterate over unique
    for entry in tqdm(pd.unique(dataValues["minMaxEntry"])):
        values = dataValues.loc[dataValues['minMaxEntry'] == entry, 'value']

        statsResult = findMinMax(entry.split(".")[0], entry.split(".")[1], entry.split(".")[2], values)
        if statsResult:
            dvPivoted.loc[entry, "min"] = statsResult["val_min"]
            dvPivoted.loc[entry, "max"] = statsResult["val_max"]
            dvPivoted.loc[entry, "comment"] = statsResult["comment"]
            dvPivoted.loc[entry, "outlier"] = statsResult["outlier"]

    print(ou, result)

    global fileResult
    if isinstance(fileResult, pd.DataFrame):
        pd.concat([fileResult, dvPivoted])
    else: 
        fileResult = dvPivoted
        

#Check if orgunit is level 1, in which case get children and do in batches
def processOrgunits(ous):

    orgunits = api.get("organisationUnits", params={
        "fields": "id,level,children[id]",
        "paging": "false",
        "filter": "id:in:[" + ",".join(ous) + "]"
    })
    orgunits = orgunits.json()["organisationUnits"]

    filtered_orgunits = []
    for orgunit in orgunits:
        if orgunit["level"] == 1:
            for child in orgunit["children"]:
                filtered_orgunits.append(child["id"])
        else:
            filtered_orgunits.append(orgunit["id"])
    
    return filtered_orgunits

for orgunit in processOrgunits(conf["orgunits"]):
    generateValues(orgunit, conf["dataset"])


if fileOutput:
    if fileOutput == "OUTLIERS":
        fileResult = fileResult[fileResult['outlier'] == True]
    
    fileName = "minMaxTest-" + conf["dataset"] + ".csv"
    fileResult.to_csv(fileName, sep='\t', encoding='utf-8')