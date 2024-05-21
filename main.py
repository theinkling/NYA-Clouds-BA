import pandas as pd
import datetime
import json
import pymannkendall as mk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import xarray as xr
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from dateutil.rrule import rrule, YEARLY, DAILY
from scipy.stats import pearsonr

# Set global  resolution for plots
dpi = 300


def radiosondePath(date):
    radiosondePath = "../Data/Radiosonde/NYA_UAS_"
    return radiosondePath + date.strftime("%Y") + ".tab"


def cloudnetPath(date):
    cloudnetPath = "../Data/Cloudnet/"
    return cloudnetPath + (
        date.strftime("%Y")
        + "/"
        + date.strftime("%Y%m%d")
        + "_ny-alesund_classification.nc"
    )


# Compute RHI factor
def RHConversionFactor(T):
    SWPWater = 610.94 * np.exp(17.625 * T / (T + 243.04))
    SWPIce = 611.21 * np.exp(22.587 * T / (T + 273.86))
    return SWPWater / SWPIce


# Max, min, inter RH thresholds
def thresholdRH(h, mode, shift=0):
    if mode == "min":
        levels = np.array([92, 90, 88, 75]) + shift
    elif mode == "max":
        levels = np.array([95, 93, 90, 80]) + shift
    else:
        levels = np.array([84, 82, 78, 70]) + shift
    if h >= 6000 and h <= 12000:
        return levels[2] - (levels[2] - levels[3]) / 6000 * (h - 6000)
    elif h < 6000 and h > 2000:
        if mode == "inter":
            return float(80.0 + shift - (80.0 - 78.0) / 4000 * (h - 2000))
        else:
            return levels[1] - (levels[1] - levels[2]) / 4000 * (h - 2000)
    elif h <= 2000:
        return levels[0] - (levels[0] - levels[1]) / 2000 * h
    elif h > 12000:
        return levels[3]


# Read CN from files
def readCloudnetData(startDate, endDate, returnCBTHOnly=False, below10KmOnly=False):
    listOfDailyData = []
    for date in tqdm(
        list(rrule(DAILY, dtstart=startDate, until=endDate)),
        desc="Reading Cloudnet Data",
    ):
        try:
            rawData = xr.open_dataset(cloudnetPath(date))
            dailyData = rawData.to_dataframe()
            rawData.close()
            dailyData.drop(
                columns=["latitude", "longitude", "altitude"], inplace=True
            )  # Drop unnecessary cols
            if below10KmOnly:
                dailyData.loc[
                    dailyData["cloud_base_height"] > 10000, "cloud_base_height"
                ] = pd.NA
                dailyData.loc[
                    dailyData["cloud_top_height"] > 10000, "cloud_top_height"
                ] = pd.NA
            if returnCBTHOnly:
                CBTH = dailyData.groupby(level="time")[
                    ["cloud_base_height", "cloud_top_height"]
                ].first()
                listOfDailyData.append(CBTH)
            else:
                listOfDailyData.append(dailyData)
        except FileNotFoundError:
            print("Warning! File not found:", cloudnetPath(date))

    return pd.concat(listOfDailyData)


# Read radiosonde data from files
def readRadiosondeData(
    startDate, endDate, preprocessData=True, onlyMiddayAscents=False
):
    dtypes = {
        "Date/Time": "object",
        "ID": "object",
        "ID": "object",
        "Altitude [m]": "int64",
        "Latitude": "float64",
        "Longitude": "float64",
        "PPPP [hPa]": "float64",
        "TTT [°C]": "float64",
        "RH [%]": "float64",
        "ff [m/s]": "float64",
        "dd [deg]": "float64",
    }
    # Read data from files
    listOfYearlyData = []
    for year in tqdm(
        list(rrule(YEARLY, dtstart=startDate, until=endDate)),
        desc="Reading Radiosonde Data",
    ):
        yearlyData = pd.read_csv(
            radiosondePath(year),
            delimiter="\t",
            skiprows=(
                range(0, 24)
                if year > datetime.datetime(2014, 12, 31, 0, 0)
                else range(0, 26)
            ),
            dtype=dtypes,
        )
        listOfYearlyData.append(yearlyData)
    data = pd.concat(listOfYearlyData)

    # Change header, convert to datetime
    data.columns = ["datetime", "id", "h", "lat", "lon", "p", "T", "rh", "ws", "wd"]
    data.drop(columns=["ws", "wd"], inplace=True)
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%dT%H:%M:%S")
    data["date"] = data["datetime"].dt.date
    # Truncate data from start to end date
    data.set_index("datetime", inplace=True)
    truncatedData = data.sort_index().truncate(
        before=startDate, after=endDate + pd.Timedelta(days=1)
    )
    # Grab radiosonde ids of all flights in that timespan
    ids = truncatedData["id"].unique()
    data.reset_index(inplace=True)
    # Set multiindex to (id, height)
    data.set_index(["id", "h"], inplace=True)
    # Use all flights with an id in the time range,
    # even if they are not entirely contained in the time range
    data = data[data.index.get_level_values("id").isin(ids)]
    # Compute RHI
    data["rhi"] = data["rh"]
    data.loc[data["T"] < 0, "rhi"] = data.loc[data["T"] < 0, "rh"] * RHConversionFactor(
        data.loc[data["T"] < 0, "T"]
    )

    # Initialize columns for cloud and moist levels for ZH10 algorithm
    data["moistZhang"] = False
    data["cloudZhang"] = False

    # Drop rows where any of the following columns are NaN
    data.dropna(subset=["rhi", "rh", "T", "datetime"], inplace=True)

    if preprocessData:
        # Drop ascent if it did not reach 10 km
        data = data.groupby(level="id").filter(
            lambda x: x.index.get_level_values("h")[-1] == 10000
        )

        # Drop ascent if duration <10 mins or >60 mins
        data = data.groupby(level="id").filter(
            lambda x: (
                x.datetime.iloc[-1] - x.datetime.iloc[0] <= pd.Timedelta(minutes=50)
            )
            and (x.datetime.iloc[-1] - x.datetime.iloc[0] >= pd.Timedelta(minutes=10))
        )
    if onlyMiddayAscents:
        data = data.groupby(level="id").filter(
            lambda x: (x.datetime.iloc[0].time() >= datetime.time(10, 0, 0))
            and (x.datetime.iloc[0].time() <= datetime.time(12, 0, 0))
        )

    return data


# Create detailed RS ascent plots
def plotRadiosondeAscentTimes(
    radiosondeData,
    xlimDur=(0, 60),
    binsDur=50,
    durFileName="./flightDurations.png",
    vlines=None,
):
    startTimes = []
    endTimes = []
    durations = []
    for id, df in radiosondeData.groupby(level="id"):
        startTimes.append(df["datetime"].iloc[0])
        endTimes.append(df["datetime"].iloc[-1])
        durations.append(df["datetime"].iloc[-1] - df["datetime"].iloc[0])
    startTimes = pd.Series(startTimes)
    endTimes = pd.Series(endTimes)
    durations = pd.Series(durations)
    startTimes = startTimes.apply(lambda x: x.replace(year=2000, month=1, day=1))
    endTimes = endTimes.apply(lambda x: x.replace(year=2000, month=1, day=1))
    mins = pd.date_range(
        start=pd.to_datetime("2000-01-01 00:00:00"),
        end=pd.to_datetime("2000-01-02 00:00:00"),
        freq="5min",
    )
    ylim = (0.5, 5 * 1000)
    plt.hist(startTimes, bins=mins)
    if vlines is not None:
        plt.axvline(
            datetime.datetime(2000, 1, 1, vlines[0]), c="red", linestyle="dashed"
        )
        plt.axvline(
            datetime.datetime(2000, 1, 1, vlines[1]), c="red", linestyle="dashed"
        )
    plt.xlabel("Uhrzeit (UTC)")
    plt.ylabel("Häufigkeit")
    plt.gca().xaxis.set_major_locator(
        mdates.HourLocator(byhour=[3 * k for k in range(0, 24)])
    )
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlim(
        pd.to_datetime("2000-01-01 00:00:00"), pd.to_datetime("2000-01-02 00:00:00")
    )
    plt.yscale("log")
    plt.ylim(ylim)
    plt.savefig("./flightStarts.png", dpi=dpi)
    plt.close()

    plt.hist(endTimes, bins=mins)
    plt.xlabel("Uhrzeit (UTC)")
    plt.ylabel("Häufigkeit")
    plt.gca().xaxis.set_major_locator(
        mdates.HourLocator(byhour=[3 * k for k in range(0, 24)])
    )
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlim(
        pd.to_datetime("2000-01-01 00:00:00"), pd.to_datetime("2000-01-02 00:00:00")
    )
    plt.yscale("log")
    plt.ylim(ylim)
    plt.savefig("./flightEnds.png", dpi=dpi)
    plt.close()

    plt.hist(durations.dt.total_seconds() / 60, bins=binsDur)
    plt.xlabel("Flugdauer [Minuten]")
    plt.ylabel("Häufigkeit")
    plt.xlim(xlimDur)
    plt.yscale("log")
    plt.ylim(ylim)
    plt.savefig(durFileName, dpi=dpi)
    plt.close()


# Zhang algorithm
def ZH10(data, shift=0):
    # The next steps only work with a single index, not a MultiIndex
    # so we reset the index to 'h'.
    data.reset_index(inplace=True)
    data.set_index("h", inplace=True)

    # Find moist altitudes
    data.loc[
        data["rhi"]
        > data.index.to_series().apply(lambda x: thresholdRH(x, "min", shift=shift)),
        "moistZhang",
    ] = True

    # Group moist layers, remove moist tags for certain conditions, and set cloud tags
    groupedDf = data.groupby(
        (data["moistZhang"] != data["moistZhang"].shift()).cumsum()
    )
    groupedDfs = []
    for group, data in groupedDf:
        if data.index[0] <= 120 and data.shape[0] < 8 and data["moistZhang"].iloc[0]:
            data["moistZhang"] = False
        if data["moistZhang"].iloc[0] and data["rhi"].max() > thresholdRH(
            data.index[0], "max", shift=shift
        ):
            data["cloudZhang"] = True
        if data["cloudZhang"].iloc[0]:
            data.loc[data.index < 280, "cloudZhang"] = False
        groupedDfs.append(data)

    combinedDf = pd.concat(groupedDfs)

    # Group cloud layers, consider different layers as same on certain condition
    groupedDf = combinedDf.groupby(
        (combinedDf["cloudZhang"] != combinedDf["cloudZhang"].shift()).cumsum()
    )
    groupedDfs = []
    for group, data in groupedDf:
        isFirstLayer = combinedDf.index[0] == data.index[0]
        if not (data["cloudZhang"].iloc[0] or isFirstLayer):
            consider_as_one = (data.index[-1] - data.index[0] < 300) or (
                data["rhi"].min() > thresholdRH(data.index[0], "inter", shift=shift)
            )
            if consider_as_one:
                data["cloudZhang"] = True
        groupedDfs.append(data)
    combinedDf = pd.concat(groupedDfs)

    # Remove cloud layers with small vertical thickness
    groupedDf = combinedDf.groupby(
        (combinedDf["cloudZhang"] != combinedDf["cloudZhang"].shift()).cumsum()
    )
    groupedDfs = []
    for group, data in groupedDf:
        if data.index[-1] - data.index[0] < 50:
            data["cloudZhang"] = False
        groupedDfs.append(data)
    combinedDf = pd.concat(groupedDfs)

    # Reset the index to (id, height)
    combinedDf.reset_index(inplace=True)
    combinedDf.set_index(["id", "h"], inplace=True)

    return combinedDf


# Call the ZH10 function for each individual
# radiosonde profile.
def callZH10(data, shift=0, showTqdm=True):
    runs = []
    for run, runData in conditionalTqdm(
        data.groupby(level="id"), showTqdm, desc="Running Zhang Algorithm"
    ):
        runs.append(ZH10(runData, shift))

    return pd.concat(runs)


# Plot RS ascent
def plotProfile(
    ZhangDf,
    path,
    rh=True,
    rhIce=True,
    scatter=False,
    rhMin=False,
    rhMax=False,
    rhInter=False,
    showThreshold=False,
    threshold=95,
    legend=True,
    xlim=(0, 150),
    ylim=(0, 10000),
):
    fig, ax = plt.subplots(figsize=(4, 5))
    """Plot RH(I) profile and threshold lines"""
    ZhangDf.reset_index(inplace=True)
    ZhangDf.set_index("h", inplace=True)
    if rh:
        if scatter:
            plt.scatter(ZhangDf["rh"], ZhangDf.index, label="RH", s=3.5)
        else:
            plt.plot(ZhangDf["rh"], ZhangDf.index, label="RH")
    if rhIce:
        if scatter:
            plt.scatter(ZhangDf["rhi"], ZhangDf.index, label="RHI", s=3.5)
        else:
            plt.plot(ZhangDf["rhi"], ZhangDf.index, label="RHI")
    if rhMax:
        plt.plot(
            ZhangDf.index.to_series().apply(lambda x: thresholdRH(x, "max")),
            ZhangDf.index,
            label="Max-RH",
        )
    if rhMin:
        plt.plot(
            ZhangDf.index.to_series().apply(lambda x: thresholdRH(x, "min")),
            ZhangDf.index,
            label="Min-RH",
        )
    if rhInter:
        plt.plot(
            ZhangDf.index.to_series().apply(lambda x: thresholdRH(x, "inter")),
            ZhangDf.index,
            label="Inter-RH",
        )

    if showThreshold:
        plt.axvline(threshold, color="grey", linestyle="--")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(km_formatter))
    plt.ylabel("Höhe [km]")
    plt.xlabel("Relative Feuchte [%]")

    checkLegendSet = False
    for cloudLayer, layerDf in ZhangDf.groupby(
        (ZhangDf["cloudZhang"] != ZhangDf["cloudZhang"].shift()).cumsum()
    ):
        if layerDf["cloudZhang"].iloc[0]:
            if checkLegendSet:
                label = None
            else:
                label = "Zhang (2010)"
                checkLegendSet = True
            rect = mpatches.Rectangle(
                (xlim[0], layerDf.index[0]),
                xlim[1] - xlim[0],
                layerDf.index[-1] - layerDf.index[0],
                linewidth=1,
                edgecolor=None,
                facecolor="darkviolet",
                alpha=0.2,
                label=label,
            )
            ax.add_patch(rect)

    # Reset index to MultiIndex
    ZhangDf.reset_index(inplace=True)
    ZhangDf.set_index(["id", "h"], inplace=True)

    if legend:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def cloudnetPlot(cloudnetData, path, radiosondeData=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    filteredCloudnetData = cloudnetData[cloudnetData["target_classification"] != 0]
    colors = [
        "deepskyblue",
        "red",
        "darkblue",
        "gold",
        "limegreen",
        "orange",
        "forestgreen",
        "lightgray",
        "darkgray",
        "dimgray",
    ]
    labels = [
        "Wolkentropfen",
        "Niesel/Regen",
        "Niesel/Regen oder Wolkentropfen",
        "Eis",
        "Eis/Unterkühlte Tropfen",
        "Schmelzendes Eis",
        "Schmelzendes Eis/Wolkentropfen",
        "Aerosole",
        "Insekten",
        "Aerosole/Insekten",
    ]
    for classificationType in range(1, 11):
        singleClassificationType = filteredCloudnetData[
            filteredCloudnetData["target_classification"] == classificationType
        ]
        plt.scatter(
            singleClassificationType.index.get_level_values("time"),
            singleClassificationType.index.get_level_values("height"),
            s=1,
            c=colors[classificationType - 1],
        )
    plt.ylim((0, 10000))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(km_formatter))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xlabel("UTC")
    plt.ylabel("Höhe [km]")
    handles = []
    for classificationType in range(1, 11):
        rect = mpatches.Patch(
            label=labels[classificationType - 1],
            edgecolor=None,
            facecolor=colors[classificationType - 1],
        )
        handles.append(rect)
    plt.plot(
        cloudnetData.index.get_level_values("time"),
        cloudnetData["cloud_base_height"],
        c="darkviolet",
    )
    handles.append(mlines.Line2D([0], [0], c="darkviolet", label="Cloud Base Height"))
    plt.plot(
        cloudnetData.index.get_level_values("time"),
        cloudnetData["cloud_top_height"],
        c="fuchsia",
    )
    handles.append(mlines.Line2D([0], [0], c="fuchsia", label="Cloud Top Height"))
    if radiosondeData is not None:
        plt.plot(radiosondeData["datetime"], radiosondeData.index, c="black")
        handles.append(mlines.Line2D([0], [0], c="black", label="Radiosonde"))
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def km_formatter(x, pos):
    return f"{int(x / 1000)}"


def cbar_formatter(value, _):
    return f"$10^{{{int(value)}}}$"


def conditionalTqdm(iterable, condition, desc=""):
    if condition:
        return tqdm(iterable, desc=desc)
    else:
        return iterable


# Create DataFrame for comparison between CN and Zhang
def createAnalysisDf(ZHData, cloudnetCBTH, showTqdm=True, removeArtefact=True):

    idList = []
    cloudnetNumberOfDataPointsList = []
    flightDurationList = []
    cloudnetCloudDetectedList = []
    cloudnetCBHMinList = []
    cloudnetCBH25List = []
    cloudnetCBHMedianList = []
    cloudnetCBH75List = []
    cloudnetCBHMaxList = []
    cloudnetCTHMinList = []
    cloudnetCTH25List = []
    cloudnetCTHMedianList = []
    cloudnetCTH75List = []
    cloudnetCTHMaxList = []
    zhangCloudDetectedList = []
    zhangCBHsList = []
    zhangCTHsList = []
    zhangNumberOfCloudLayersList = []
    dropped = 0

    # Loop over single runs and append the analysis data as a row
    for run, runData in conditionalTqdm(
        ZHData.groupby(level="id"), showTqdm, desc="Analyzing data"
    ):
        # Add id of radiosonde ascent to list
        idList.append(run)

        # Only use Cloudnet data during the radiosonde ascent
        cloudnetCBTHDuringAscent = cloudnetCBTH.truncate(
            before=runData["datetime"].iloc[0], after=runData["datetime"].iloc[-1]
        )

        # Compute quartiles of Cloudnet CBH and CTH
        (
            cloudnetCBHMin,
            cloudnetCBH25,
            cloudnetCBHMedian,
            cloudnetCBH75,
            cloudnetCBHMax,
        ) = cloudnetCBTHDuringAscent["cloud_base_height"].quantile(
            [0, 0.25, 0.5, 0.75, 1]
        )
        (
            cloudnetCTHMin,
            cloudnetCTH25,
            cloudnetCTHMedian,
            cloudnetCTH75,
            cloudnetCTHMax,
        ) = cloudnetCBTHDuringAscent["cloud_top_height"].quantile(
            [0, 0.25, 0.5, 0.75, 1]
        )
        cloudnetCBHMinList.append(cloudnetCBHMin)
        cloudnetCBH25List.append(cloudnetCBH25)
        cloudnetCBHMedianList.append(cloudnetCBHMedian)
        cloudnetCBH75List.append(cloudnetCBH75)
        cloudnetCBHMaxList.append(cloudnetCBHMax)
        cloudnetCTHMinList.append(cloudnetCTHMin)
        cloudnetCTH25List.append(cloudnetCTH25)
        cloudnetCTHMedianList.append(cloudnetCTHMedian)
        cloudnetCTH75List.append(cloudnetCTH75)
        cloudnetCTHMaxList.append(cloudnetCTHMax)

        # Cloudnet detects a cloud if there is any non-NaN entry during the radiosonde ascent
        cloudnetCloudDetected = (
            cloudnetCBTHDuringAscent["cloud_base_height"].notnull()
            & cloudnetCBTHDuringAscent["cloud_top_height"].notnull()
        ).any()
        cloudnetCloudDetectedList.append(cloudnetCloudDetected)

        # Extract all CBHs and CTHs from Zhang detected clouds
        groupedDf = runData.groupby(
            (runData["cloudZhang"] != runData["cloudZhang"].shift()).cumsum()
        )
        zhangCBHs, zhangCTHs = [], []
        for layer, layerDf in groupedDf:
            if layerDf["cloudZhang"].iloc[0]:
                zhangCBHs.append(layerDf.index.get_level_values("h")[0])
                zhangCTHs.append(layerDf.index.get_level_values("h")[-1])
        if removeArtefact:
            dropped += zhangCTHs.count(10000)
            zhangCTHs = [i for i in zhangCTHs if i != 10000]

        # Zhang detects a cloud if there is any layer with Zhang cloud detected
        zhangCloudDetected = len(zhangCTHs) > 0
        zhangCloudDetectedList.append(zhangCloudDetected)

        zhangCBHsList.append(zhangCBHs)
        zhangCTHsList.append(zhangCTHs)

        # Compute number of cloud layers detected by zhang
        zhangNumberOfCloudLayersList.append(len(zhangCTHs))

        # Compute number of Cloudnet data points during ascent
        cloudnetNumberOfDataPointsList.append(len(cloudnetCBTHDuringAscent))

        # Compute flight duration
        flightDurationList.append(
            (runData["datetime"].iloc[-1] - runData["datetime"].iloc[0])
        )
    print(str(dropped) + " occurences of 10km dropped!")
    # Create dataframe holding all the lists
    analysisData = pd.DataFrame(
        {
            "id": idList,
            "cloudnetNumberOfDataPoints": cloudnetNumberOfDataPointsList,
            "flightDuration": flightDurationList,
            "cloudnetCloudDetected": cloudnetCloudDetectedList,
            "cloudnetCBHMin": cloudnetCBHMinList,
            "cloudnetCBH25": cloudnetCBH25List,
            "cloudnetCBHMedian": cloudnetCBHMedianList,
            "cloudnetCBH75": cloudnetCBH75List,
            "cloudnetCBHMax": cloudnetCBHMaxList,
            "cloudnetCTHMin": cloudnetCTHMinList,
            "cloudnetCTH25": cloudnetCTH25List,
            "cloudnetCTHMedian": cloudnetCTHMedianList,
            "cloudnetCTH75": cloudnetCTH75List,
            "cloudnetCTHMax": cloudnetCTHMaxList,
            "zhangCloudDetected": zhangCloudDetectedList,
            "zhangCBHs": zhangCBHsList,
            "zhangCTHs": zhangCTHsList,
            "zhangNumberOfCloudLayers": zhangNumberOfCloudLayersList,
        }
    )

    # Compute density of Cloudnet data points per second
    analysisData["cloudnetDataDensity"] = (
        analysisData.cloudnetNumberOfDataPoints
        / analysisData.flightDuration.apply(lambda x: x.total_seconds() / (60 * 60))
    )
    return analysisData


# Plot CN data density during RS ascent
def plotCloudnetDataDensity(analysisData, cloudnetDataDensityThreshold):
    (analysisData["cloudnetDataDensity"] * 100 / 120).plot(
        kind="hist",
        bins=100,
        xlabel="Datendichte [%] ",
        ylabel="Häufigkeit",
    )
    plt.yscale("log")
    plt.axvline(100 * cloudnetDataDensityThreshold / 120, c="red", linestyle="dashed")
    plt.savefig("./cloudnetDataDensity.png", dpi=dpi)
    plt.close()


# Compute Skill Scores
def createContingencyTableAndSkillScores(analysisData, cloudnetDataDensityThreshold):
    filteredAnalysisData = analysisData[
        analysisData["cloudnetDataDensity"] >= cloudnetDataDensityThreshold
    ]

    contingencyTable = {
        "nonDiscarded": len(filteredAnalysisData),
        "discarded": len(analysisData) - len(filteredAnalysisData),
        "Korrekt Positiv": len(
            filteredAnalysisData[
                filteredAnalysisData.cloudnetCloudDetected
                & filteredAnalysisData.zhangCloudDetected
            ]
        ),
        "Korrekt Negativ": len(
            filteredAnalysisData[
                ~(
                    filteredAnalysisData.cloudnetCloudDetected
                    | filteredAnalysisData.zhangCloudDetected
                )
            ]
        ),
        "Falsch Positiv": len(
            filteredAnalysisData[
                (~filteredAnalysisData.cloudnetCloudDetected)
                & filteredAnalysisData.zhangCloudDetected
            ]
        ),
        "Falsch Negativ": len(
            filteredAnalysisData[
                filteredAnalysisData.cloudnetCloudDetected
                & (~filteredAnalysisData.zhangCloudDetected)
            ]
        ),
    }
    a = contingencyTable["Korrekt Positiv"]
    b = contingencyTable["Falsch Positiv"]
    c = contingencyTable["Falsch Negativ"]
    d = contingencyTable["Korrekt Negativ"]
    T = contingencyTable["nonDiscarded"]
    # Overall, what fraction of the forecasts were correct?
    contingencyTable["Accuracy"] = (a + d) / T
    # What fraction of the observed "yes" events were correctly forecast?
    contingencyTable["Hit Rate"] = a / (a + c)
    # What fraction of the observed "no" events were incorrectly forecast as "yes"?
    contingencyTable["False Alarm Rate"] = b / (d + b)
    # What fraction of the forecast "yes" events were correctly observed?
    contingencyTable["Success Ratio"] = a / (a + b)
    # What fraction of the predicted "yes" events actually did not occur (i.e., were false alarms)?
    contingencyTable["False Alarm Ratio"] = b / (a + b)
    # How did the forecast frequency of "yes" events compare to the observed frequency of "yes" events?
    contingencyTable["Frequency Bias"] = (a + b) / (a + c)
    # What was the accuracy of the forecast relative to that of random chance?
    contingencyTable["HSS"] = (
        (a + d) - ((a + b) * (a + c) + (c + d) * (b + d)) / T
    ) / (T - (((a + b) * (a + c) + (c + d) * (b + d)) / T))
    # How well did the forecast "yes" events correspond to the observed "yes" events?
    contingencyTable["Threat Score"] = a / (a + b + c)
    # How well did the forecast "yes" events correspond to the observed "yes" events (accounting for hits due to chance)?
    contingencyTable["ETS"] = (a - (a + b) * (a + c) / T) / (
        a + b + c - (a + b) * (a + c) / T
    )
    # What is the ratio of the odds of a "yes" forecast being correct, to the odds of a "yes" forecast being wrong? Then log of that.
    try:
        contingencyTable["Log(OR)"] = (a * d) / (b * c)
    except ZeroDivisionError:
        if a * d == b * c:
            contingencyTable["Log(OR)"] = np.nan
        else:
            contingencyTable["Log(OR)"] = np.inf
    contingencyTable["Log(OR)"] = np.log(contingencyTable["Log(OR)"])
    # What was the improvement of the forecast over random chance?
    contingencyTable["OR Skill Score"] = (a * d - b * c) / (a * d + b * c)
    # How well did the forecast separate the "yes" events from the "no" events?
    contingencyTable["PSS"] = (
        contingencyTable["Hit Rate"] - contingencyTable["False Alarm Ratio"]
    )
    contingencyTable["CSS"] = (a * d - b * c) / ((a + b) * (c + d))
    return contingencyTable


# Case studies plot
def createRunPlots(ZHData, cloudnetData):
    for run, runData in tqdm(ZHData.groupby(level="id"), desc="Creating plots"):
        plotProfile(
            runData,
            "../plots/RS"
            + runData["datetime"].iloc[0].strftime("%Y-%m-%d")
            + "-"
            + run
            + ".png",
        )

        cloudnetPlot(
            cloudnetData.loc[
                (
                    slice(
                        runData["datetime"].iloc[0] - pd.Timedelta(minutes=30),
                        runData["datetime"].iloc[-1] + pd.Timedelta(minutes=30),
                    ),
                    slice(None),
                    slice(None),
                )
            ],
            "../plots/CN"
            + runData["datetime"].iloc[0].strftime("%Y-%m-%d")
            + "-"
            + run
            + ".png",
            ZHData.loc[run],
        )


# Run shifted Zhang algorithm
def zhangSensitivityAnalysis(
    startDate,
    endDate,
    ZHData,
    radiosondeData,
    cloudnetCBTH,
    shifts,
    cloudnetDataDensityThreshold,
    saveData=False,
):
    contingencyTables = dict()
    for shift in tqdm(shifts, desc="Running shifted Zhang algorithm"):
        ZHData = callZH10(radiosondeData, shift, showTqdm=False)
        contingencyTables[float(shift)] = createContingencyTableAndSkillScores(
            createAnalysisDf(ZHData, cloudnetCBTH, showTqdm=False),
            cloudnetDataDensityThreshold,
        )
    if saveData:
        with open("./contingencyTables.json", "w") as jsonFile:
            json.dump(contingencyTables, jsonFile)
    return contingencyTables


# Plot results of sensitivity analysis
def plotSkillScores(contingencyTables):
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    skillScores = list(contingencyTables["7.0"].keys())
    skillScores.remove("nonDiscarded")
    skillScores.remove("discarded")
    skillScores.remove("Success Ratio")
    shifts = list(contingencyTables.keys())
    for skillScore, i in zip(skillScores, range(0, len(skillScores))):
        values = [contingencyTables[shift][skillScore] for shift in shifts]
        axs[i // 4][i % 4].plot([float(shift) for shift in shifts], values)
        axs[i // 4][i % 4].set_title(skillScore)
    # plt.show()
    plt.savefig("./sensitivityAnalysis.png", dpi=dpi)
    plt.close()


# Plot correlation between CN and Zhang
def plotCorrelations(analysisData, cbhFileName="", cthFileName=""):
    figsize = (6, 6)
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(
        analysisData["cloudnetCBHMedian"] / 1000,
        analysisData["cloudnetCBHMedian"] / 1000,
        linewidth=0.6,
        color="darkgrey",
    )
    scatterCBH = ax1.scatter(
        analysisData["cloudnetCBHMedian"] / 1000,
        analysisData["zhangCBHs"].apply(lambda x: min(x, default=np.nan)) / 1000,
        s=2,
        zorder=2.5,
        c=np.log10(analysisData["cloudnetCBHMax"] - analysisData["cloudnetCBHMin"]),
    )
    ax1.set_xlabel("CloudNet CBH [km]")
    ax1.set_ylabel("Zhang CBH [km]")
    ax1.set_ylim((0, 10.5))
    ax1.set_xlim((0, 10.5))
    fig1.savefig(cbhFileName + ".png", dpi=dpi)
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(
        analysisData["cloudnetCBHMedian"] / 1000,
        analysisData["cloudnetCBHMedian"] / 1000,
        linewidth=0.6,
        color="darkgrey",
    )
    ax2.scatter(
        analysisData["cloudnetCTHMedian"] / 1000,
        analysisData["zhangCTHs"].apply(lambda x: max(x, default=np.nan)) / 1000,
        s=2,
        zorder=2.5,
        c=np.log10(analysisData["cloudnetCTHMax"] - analysisData["cloudnetCTHMin"]),
    )
    ax2.set_xlabel("CloudNet CTH [km]")
    ax2.set_ylabel("Zhang CTH [km]")
    ax2.set_ylim((0, 10.5))
    ax2.set_xlim((0, 10.5))
    fig2.savefig(cthFileName + ".png", dpi=dpi)
    fig3, ax3 = plt.subplots(figsize=(6, 20))
    cbar = fig3.colorbar(scatterCBH, ax=ax3)
    cbar.set_label("Max - Min")
    cbar.set_ticks(np.unique(cbar.get_ticks().astype(int))[1:-1])
    cbar.set_ticklabels(["$10^{{{}}}$".format(int(t)) + " m" for t in cbar.get_ticks()])
    fig3.savefig("colorbar.png", dpi=dpi)
    plt.close()


# Calculate corr coefficients
def calculateCorrelation(analysisData):
    correlationParameters = dict()
    analysisDataWithoutNaN = analysisData.copy()
    analysisDataWithoutNaN.zhangCBHs = analysisDataWithoutNaN.zhangCBHs.apply(
        lambda x: min(x, default=np.nan)
    )
    analysisDataWithoutNaN.zhangCTHs = analysisDataWithoutNaN.zhangCTHs.apply(
        lambda x: max(x, default=np.nan)
    )

    analysisDataWithoutNaN.dropna(
        subset=["zhangCBHs", "zhangCTHs", "cloudnetCBHMedian", "cloudnetCTHMedian"],
        inplace=True,
    )

    correlationParameters["corrCBH"], correlationParameters["pValueCBH"] = pearsonr(
        analysisDataWithoutNaN.cloudnetCBHMedian, analysisDataWithoutNaN.zhangCBHs
    )
    correlationParameters["corrCTH"], correlationParameters["pValueCTH"] = pearsonr(
        analysisDataWithoutNaN.cloudnetCTHMedian, analysisDataWithoutNaN.zhangCTHs
    )
    correlationParameters["biasCBH"] = (1 / len(analysisDataWithoutNaN)) * (
        analysisDataWithoutNaN.cloudnetCBHMedian - analysisDataWithoutNaN.zhangCBHs
    ).sum()
    correlationParameters["biasCTH"] = (1 / len(analysisDataWithoutNaN)) * (
        analysisDataWithoutNaN.cloudnetCTHMedian - analysisDataWithoutNaN.zhangCTHs
    ).sum()
    correlationParameters["rmsdCBHFormula"] = np.sqrt(
        (1 / len(analysisDataWithoutNaN))
        * np.square(
            (
                analysisDataWithoutNaN.cloudnetCBHMedian
                - analysisDataWithoutNaN.zhangCBHs
            )
        ).sum()
    )
    correlationParameters["rmsdCTHFormula"] = np.sqrt(
        (1 / len(analysisDataWithoutNaN))
        * np.square(
            (
                analysisDataWithoutNaN.cloudnetCTHMedian
                - analysisDataWithoutNaN.zhangCTHs
            )
        ).sum()
    )
    correlationParameters["stdCTH"] = np.sqrt(
        1
        / len(analysisDataWithoutNaN)
        * np.square(
            analysisDataWithoutNaN.cloudnetCTHMedian
            - analysisDataWithoutNaN.zhangCTHs
            - correlationParameters["biasCTH"]
        ).sum()
    )
    correlationParameters["stdCBH"] = np.sqrt(
        1
        / len(analysisDataWithoutNaN)
        * np.square(
            analysisDataWithoutNaN.cloudnetCBHMedian
            - analysisDataWithoutNaN.zhangCBHs
            - correlationParameters["biasCBH"]
        ).sum()
    )
    return correlationParameters


# Plot total RS ascents
def plotRSTimelines():
    plt.figure(figsize=(10, 8))

    def computeYearCounts(radiosondeData):
        plottingData = radiosondeData.reset_index().copy()
        plottingData.drop_duplicates("id", inplace=True)
        plottingData["year"] = plottingData["datetime"].dt.year
        yearCounts = plottingData.groupby("year").size()
        return yearCounts

    totalStart = datetime.date(1993, 1, 1)
    totalEnd = datetime.date(2022, 12, 31)
    total = computeYearCounts(
        readRadiosondeData(
            totalStart, totalEnd, preprocessData=False, onlyMiddayAscents=False
        )
    )
    lowQuality = total - computeYearCounts(
        readRadiosondeData(
            totalStart, totalEnd, preprocessData=True, onlyMiddayAscents=False
        )
    )
    middayOnly = computeYearCounts(
        readRadiosondeData(
            totalStart, totalEnd, preprocessData=True, onlyMiddayAscents=True
        )
    )
    plt.bar(
        middayOnly.index,
        middayOnly,
        label="12 UTC",
        color="darkgreen",
    )
    plt.bar(
        (total - lowQuality - middayOnly).index,
        total - lowQuality - middayOnly,
        bottom=middayOnly,
        label="Nicht 12 UTC",
        color="goldenrod",
    )
    plt.bar(
        lowQuality.index,
        lowQuality,
        bottom=total - lowQuality,
        label="Verworfen",
        color="firebrick",
    )
    plt.ylabel("Anzahl")
    plt.xlabel("Jahr")
    plt.axhline(y=365, color="dimgrey", linestyle="--")
    plt.legend(loc="upper left")
    plt.xlim(1990, 2025)
    plt.xticks(range(1990, 2025, 5))
    plt.savefig("./radiosondeAscentSplit.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# Plot timeline of different RS models
def plotRSModels():
    plt.figure(figsize=(10, 2))
    positions = np.arange(1, 5, 1) / 2
    modelNames = ["RS80-A", "RS90", "RS92", "RS41"]
    usageStart = [
        datetime.date(1993, 1, 1),
        datetime.date(2002, 7, 23),
        datetime.date(2006, 5, 20),
        datetime.date(2015, 3, 1),
    ]

    usageEnd = [
        datetime.date(2003, 1, 31),
        datetime.date(2007, 1, 17),
        datetime.date(2018, 3, 31),
        datetime.date(2022, 12, 31),
    ]

    durations = [(end - start).days for start, end in zip(usageStart, usageEnd)]
    plt.barh(y=positions, width=durations, left=usageStart, height=0.4)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.yticks(positions, labels=modelNames)
    plt.xlabel("Jahr")
    plt.xlim(datetime.date(1990, 1, 1), datetime.date(2025, 1, 1))
    plt.xticks(
        pd.date_range(
            start=datetime.date(1990, 1, 1), end=datetime.date(2025, 1, 1), freq="5YE"
        )
    )
    plt.savefig("./RSModels.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# Plot data availability of CN over time
def plotCNDataAvailability():
    plt.figure(figsize=(10, 3))
    cloudnetCBTH = pd.read_csv("../Data/c.csv")
    cloudnetCBTH["time"] = pd.to_datetime(
        cloudnetCBTH["time"], format="%Y-%m-%d %H:%M:%S.%f"
    )
    cloudnetCBTH.set_index("time", inplace=True)
    dailyCounts = cloudnetCBTH.resample("D").size() / (2 * 60 * 24 / 100)
    plt.scatter(dailyCounts.index, dailyCounts, s=0.8, c="black")
    plt.xlabel("Jahr")
    plt.ylabel("Datenverfügbarkeit [%]")
    plt.savefig("./CNDataAvailability.png", dpi=dpi, bbox_inches="tight")
    plt.close()


# Compute MK and plot MK line, for the total interval and the two sub-intervals
def plotMannKendall(data, col_name, modString="", ylims=None, ylabel="", scale=1000):
    firstInterval = pd.date_range(
        data.index[0],
        data.index[0] + (2 / 3) * (data.index[-1] - data.index[0]),
        freq="D",
    )
    secondInterval = pd.date_range(
        data.index[-1] - (2 / 3) * (data.index[-1] - data.index[0]),
        data.index[-1],
        freq="D",
    )
    plt.scatter(data.index, data[col_name] / scale, s=0.5)

    mannkendall = mk.original_test(data[col_name] / scale)
    mk1 = mk.original_test(data[col_name].reindex(firstInterval) / scale)
    mk2 = mk.original_test(data[col_name].reindex(secondInterval) / scale)
    print("MK " + col_name + modString + " combined:")
    print(mannkendall)
    print("MK " + col_name + modString + " Interval 1:")
    print(mk1)
    print("MK " + col_name + modString + " Interval 2:")
    print(mk2)
    plt.plot(
        firstInterval,
        mk1.intercept + mk1.slope * (firstInterval - firstInterval[0]).days,
        c="limegreen",
        alpha=0.8,
    )
    plt.plot(
        secondInterval,
        mk2.intercept + mk2.slope * (secondInterval - secondInterval[0]).days,
        c="limegreen",
        alpha=0.8,
    )
    plt.plot(
        data.index,
        mannkendall.intercept + mannkendall.slope * (data.index - data.index[0]).days,
        c="red",
    )
    plt.xlabel("Jahr")
    if ylims is not None:
        plt.ylim(ylims)
    plt.ylabel(ylabel)
    plt.savefig("./mk" + col_name + modString + ".png", dpi=dpi)
    plt.close()


# Plot RS flights
def plotLatLon(rsData, pltName=""):
    for id, data in rsData.groupby("id"):
        plt.plot(data["lon"], data["lat"], lw=0.2)
    plt.ylim((78, 80))
    plt.xlabel("Längengrad")
    plt.ylabel("Breitengrad")
    plt.axis("equal")
    plt.savefig(pltName + ".png", dpi=dpi)
    plt.close()


# Plot RS flight deviation
def plotLatLonDistribution(rsData):
    medianLats = (rsData["lat"] - 78.923010).abs().groupby(level="h").median()
    q25Lats = (rsData["lat"] - 78.923010).abs().groupby(level="h").quantile(q=0.25)
    q75Lats = (rsData["lat"] - 78.923010).abs().groupby(level="h").quantile(q=0.75)
    medianLons = (rsData["lon"] - 11.922710).abs().groupby(level="h").median()
    q25Lons = (rsData["lon"] - 11.922710).abs().groupby(level="h").quantile(q=0.25)
    q75Lons = (rsData["lon"] - 11.922710).abs().groupby(level="h").quantile(q=0.75)
    plt.plot(medianLats, medianLats.index / 1000, label="Breitengrad")
    plt.plot(medianLons, medianLons.index / 1000, label="Längengrad")
    plt.gca().fill_betweenx(medianLats.index / 1000, q25Lats, q75Lats, alpha=0.2)
    plt.gca().fill_betweenx(medianLons.index / 1000, q25Lons, q75Lons, alpha=0.2)
    plt.xlabel("Verdriftung [°]")
    plt.ylabel("Höhe [km]")
    plt.legend()
    plt.savefig("latlonDist.png", dpi=dpi)
    plt.close()


# -------       Parameters      --------------------------
# Data points per hour needed for CN to be a valid source of comparison
cloudnetDataDensityThreshold = 105


# Create radiosonde models plot
# plotRSModels()

# Create histogram of radiosonde ascents
# plotRSTimelines()

# Create plots for radiosonde ascent times and durations
# RSData = readRadiosondeData(
#     datetime.date(1993, 1, 1),
#     datetime.date(2022, 12, 31),
#     preprocessData=True,
#     onlyMiddayAscents=False,
# )
# plotRadiosondeAscentTimes(RSData, vlines=[10, 12])

# Plot drift of radiosondes
# plotLatLon(RSData, "LatLon")
# plotLatLonDistribution(RSData)

# Plot CN data availability
# plotCNDataAvailability()

# Compute Zhang algorithm for RS data
# RSData = readRadiosondeData(
#     datetime.date(2016, 6, 10),
#     datetime.date(2022, 12, 31),
#     preprocessData=True,
#     onlyMiddayAscents=False,
# )
# ZHData = callZH10(RSData, shift=0)

# Compute CN data density for RS ascents
# cloudnetCBTHs = readCloudnetData(
#     datetime.date(2016, 6, 10),
#     datetime.date(2022, 12, 31),
#     below10KmOnly=True,
#     returnCBTHOnly=True,
# )
# cloudnetCBTHs.to_csv("../Data/c.csv")
# cloudnetCBTHs = pd.read_csv("../Data/c.csv")
# cloudnetCBTHs["time"] = pd.to_datetime(
#     cloudnetCBTHs["time"], format="%Y-%m-%d %H:%M:%S.%f"
# )
# cloudnetCBTHs.set_index("time", inplace=True)

# plotCloudnetDataDensity(
#     createAnalysisDf(ZHData, cloudnetCBTHs, removeArtefact=False),
#     cloudnetDataDensityThreshold,
# )

# Compute Skill Scores and contingency tables for cloud detection
# analysisData = createAnalysisDf(ZHData, cloudnetCBTHs, removeArtefact=False)
# cT0 = createContingencyTableAndSkillScores(analysisData, cloudnetDataDensityThreshold)
# print("Contingency table and Skill Scores for unshifted ZhA:")
# print(cT0)

# Create sensitivity plots
# contingencyTables = zhangSensitivityAnalysis(
#     datetime.date(2016, 6, 10),
#     datetime.date(2022, 12, 31),
#     ZHData,
#     RSData,
#     cloudnetCBTHs,
#     np.arange(-20, 20, 1),
#     cloudnetDataDensityThreshold,
#     saveData=True,
# )
# with open("contingencyTables.json", "r") as file:
#     contingencyTables = json.load(file)
# plotSkillScores(contingencyTables)

# Compute Skill Scores and contingency tables for cloud detection with shifted ZhA
# analysisData7 = createAnalysisDf(
#     callZH10(RSData, shift=7), cloudnetCBTHs, removeArtefact=False
# )
# cT7 = createContingencyTableAndSkillScores(analysisData7, cloudnetDataDensityThreshold)
# print("Contingency table and Skill Scores for shifted ZhA:")
# print(cT7)

# Plot correlations
# analysisData0A = createAnalysisDf(
#     callZH10(RSData, shift=0), cloudnetCBTHs, removeArtefact=True
# )
# analysisData7A = createAnalysisDf(
#     callZH10(RSData, shift=7), cloudnetCBTHs, removeArtefact=True
# )
# plotCorrelations(analysisData, cbhFileName="cbhC", cthFileName="cthC")
# plotCorrelations(analysisData7, cbhFileName="cbhC7", cthFileName="cthC7")
# plotCorrelations(analysisData0A, cbhFileName="cbhC0A", cthFileName="cthC0A")
# plotCorrelations(analysisData7A, cbhFileName="cbhC7A", cthFileName="cthC7A")


# Print Correlation parameters
# for aD, name in zip(
#     [analysisData, analysisData0A, analysisData7, analysisData7A],
#     ["0", "0A", "7", "7A"],
# ):
#     correlationCoefficients = calculateCorrelation(aD)
#     print(name)
#     for i in range(0, len(correlationCoefficients)):
#         print(
#             list(correlationCoefficients.keys())[i]
#             + ":\t"
#             + str(list(correlationCoefficients.values())[i])
#         )
#         print()


# --------------    Create Radiosonde Ascent Plots and Cloudnet Plots   --------------------
# radiosondeData = readRadiosondeData(startDate, endDate, preprocessData=True)
# ZHData = callZH10(radiosondeData, shift=0)
# cloudnetData = readCloudnetData(startDate, endDate)
# createRunPlots(ZHData, cloudnetData)

# ---------------   Read and prepare data for trend analysis   ----------------------------
radiosondeDataAllTime = readRadiosondeData(
    datetime.date(1993, 1, 1),
    datetime.date(2022, 12, 31),
    preprocessData=True,
    onlyMiddayAscents=True,
)

# ------------  Keep only the last run if several ascents in one day    ----------------------
ascentStarts = (
    radiosondeDataAllTime.groupby(level="id")
    .first()
    .reset_index()
    .sort_values(by=["datetime"])
    .set_index("id")
)
radiosondeDataAllTime.drop(
    ascentStarts[ascentStarts.duplicated(subset=["date"], keep="last")].index,
    inplace=True,
)

# Run ZhA algorithm
ZHData = callZH10(radiosondeDataAllTime)
ZH7Data = callZH10(radiosondeDataAllTime, shift=7)


# Extract CBH and CTH from ZhA
def extractCBTHs(ZHDf):
    idList = []
    zhangCBHList = []
    zhangCTHList = []
    dateList = []
    dropped = 0
    for run, runData in tqdm(
        ZHDf.groupby(level="id"), desc="Extracting CBH and CTH from Zhang data"
    ):
        idList.append(run)
        dateList.append(runData.date.iloc[0])
        zhangCBHs = []
        zhangCTHs = []
        groupedDf = runData.groupby(
            (runData["cloudZhang"] != runData["cloudZhang"].shift()).cumsum()
        )
        for _, layerDf in groupedDf:
            if layerDf["cloudZhang"].iloc[0]:
                zhangCBHs.append(layerDf.index.get_level_values("h")[0])
                zhangCTHs.append(layerDf.index.get_level_values("h")[-1])

        # Use second highest CTH if highest is 10km
        dropped += zhangCTHs.count(10000)
        zhangCTHs = [i for i in zhangCTHs if i != 10000]

        zhangCBHList.append(min(zhangCBHs, default=np.nan))
        zhangCTHList.append(max(zhangCTHs, default=np.nan))

    zhangCBTHs = pd.DataFrame(
        {"id": idList, "date": dateList, "CBH": zhangCBHList, "CTH": zhangCTHList},
    )
    zhangCBTHs["date"] = pd.to_datetime(zhangCBTHs["date"])
    zhangCBTHs.set_index("date", inplace=True)
    zhangCBTHs.sort_index(inplace=True)
    zhangCBTHs.drop("id", axis=1, inplace=True)
    print(str(dropped) + " occurences above 10 km dropped!")
    return zhangCBTHs


zhangCBTHs = extractCBTHs(ZHData)
zhang7CBTHs = extractCBTHs(ZH7Data)

# Plot MK trends

plotMannKendall(
    zhangCBTHs.reindex(
        pd.date_range(zhangCBTHs.index[0], zhangCBTHs.index[-1], freq="D")
    ),
    "CBH",
    "0",
    ylims=(0, 10.5),
    ylabel="h [km]",
)
plotMannKendall(
    zhangCBTHs.reindex(
        pd.date_range(zhangCBTHs.index[0], zhangCBTHs.index[-1], freq="D")
    ),
    "CTH",
    "0",
    ylims=(0, 10.5),
    ylabel="h [km]",
)
plotMannKendall(
    zhang7CBTHs.reindex(
        pd.date_range(zhang7CBTHs.index[0], zhang7CBTHs.index[-1], freq="D")
    ),
    "CBH",
    "7",
    ylims=(0, 10.5),
    ylabel="h [km]",
)
plotMannKendall(
    zhang7CBTHs.reindex(
        pd.date_range(zhang7CBTHs.index[0], zhang7CBTHs.index[-1], freq="D")
    ),
    "CTH",
    "7",
    ylims=(0, 10.5),
    ylabel="h [km]",
)

# MK for normalized data
z = zhangCBTHs.copy()
z.CBH, z.CTH = pd.NA, pd.NA
zhangCBTHsMonthly = zhangCBTHs.resample("ME").mean()
zhangCBTHsMonthlyNaN = z.combine_first(zhangCBTHsMonthly)
zhangCBTHsMonthly["m"] = zhangCBTHsMonthly.index.month
monthlyMeans = zhangCBTHsMonthly.groupby("m").transform("mean")
monthlyStds = zhangCBTHsMonthly.groupby("m").transform("std")
zhangCBTHsMonthlyNaN["CBHm"] = zhangCBTHsMonthlyNaN["CBH"] - monthlyMeans["CBH"]
zhangCBTHsMonthlyNaN["CTHm"] = zhangCBTHsMonthlyNaN["CTH"] - monthlyMeans["CTH"]
zhangCBTHsMonthlyNaN["CBHms"] = zhangCBTHsMonthlyNaN["CBHm"] / monthlyStds["CBH"]
zhangCBTHsMonthlyNaN["CTHms"] = zhangCBTHsMonthlyNaN["CTHm"] / monthlyStds["CTH"]

for col_name in ["CBHms", "CTHms"]:
    plotMannKendall(
        zhangCBTHsMonthlyNaN, col_name, "0", ylims=(-3.5, 3.5), ylabel="z", scale=1
    )
for col_name in ["CBHm", "CTHm"]:
    plotMannKendall(
        zhangCBTHsMonthlyNaN, col_name, "0", ylims=(-3, 3), ylabel="Δh [km]"
    )

z = zhang7CBTHs.copy()
z.CBH, z.CTH = pd.NA, pd.NA
zhang7CBTHsMonthly = zhang7CBTHs.resample("ME").mean()
zhang7CBTHsMonthlyNaN = z.combine_first(zhang7CBTHsMonthly)
zhang7CBTHsMonthly["m"] = zhang7CBTHsMonthly.index.month
monthlyMeans = zhang7CBTHsMonthly.groupby("m").transform("mean")
monthlyStds = zhang7CBTHsMonthly.groupby("m").transform("std")
zhang7CBTHsMonthlyNaN["CBHm"] = zhang7CBTHsMonthlyNaN["CBH"] - monthlyMeans["CBH"]
zhang7CBTHsMonthlyNaN["CTHm"] = zhang7CBTHsMonthlyNaN["CTH"] - monthlyMeans["CTH"]
zhang7CBTHsMonthlyNaN["CBHms"] = zhang7CBTHsMonthlyNaN["CBHm"] / monthlyStds["CBH"]
zhang7CBTHsMonthlyNaN["CTHms"] = zhang7CBTHsMonthlyNaN["CTHm"] / monthlyStds["CTH"]

for col_name in ["CBHms", "CTHms"]:
    plotMannKendall(
        zhang7CBTHsMonthlyNaN, col_name, "7", ylims=(-3.5, 3.5), ylabel="z", scale=1
    )
for col_name in ["CBHm", "CTHm"]:
    plotMannKendall(
        zhang7CBTHsMonthlyNaN, col_name, "7", ylims=(-3, 3), ylabel="Δh [km]"
    )
