import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from SDLogGeneration import SDLogGeneration
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_model import ARIMA


class TWSelection:

    # for the Arrival rate check the existing patterns inside the different SD-Logs
    def Detect_pattern_tw(self, Overall_dict, event_log,firsttime):
        # TODO  Find suitable  bin  for time

        # Todo Read features from SD: dict

        TW_Dete_dict = defaultdict(list)
        number_figure = round(math.sqrt(len(Overall_dict["Arrival rate"].keys())))
        if firsttime==True:
            vid = 1
            plt.figure()
            for ktw, vtw in Overall_dict["Arrival rate"].items():
                plt.rcParams["figure.figsize"] = [10, 10]
                plt.subplot(2, 2, vid)
                plt.tight_layout()
                plt.gca().set_title("Arrival Rate Per " + str(ktw), size=8)

               #vtwn = (vtw - np.min(vtw)) / (np.max(vtw) - np.min(vtw))
                plt.plot([i for i in range(0, len(vtw))], vtw,'--ob', label=str(ktw))
                vid = vid + 1

            # plt.savefig('/UserPattern.png', dpi=100)
            plt.show()
        # plt.bar([i for i in range(0, len(vtw))], vtw)
        # plt.show()

        # TODO Detrend by removing difference:
        # diff = list()
        # for i in range(1, len(vtw)):
        #   value = vtw[i] - vtw[i - 1]
        #  diff.append(value)
        # plt.plot(diff)
        # plt.show()

        # ،TODO Find lag with maximum Corr and best TW:
        for ktw, vtw in Overall_dict["Arrival rate"].items():
            if len(vtw)>2:
                max_lag = (len(vtw) // 10)
                if max_lag > 0:

                    plt.figure()
                    temp_acorr = plt.acorr(np.array(vtw).astype(float), maxlags=max_lag)
                    temp_acorr_1 = plt.acorr(np.array(vtw).astype(float), maxlags=max_lag)
                    # temp_acorr = plt.acorr(np.array(vtw).astype(float))
                    temp_acorr[1][temp_acorr[1] == 1.0] = 0
                    TW_Dete_dict[ktw].append(np.max(temp_acorr[1]))
                    index_max = np.argmax(temp_acorr[1])
                    TW_Dete_dict[ktw].append(temp_acorr[0][index_max])
                    TW_Dete_dict[ktw].append(temp_acorr[1][np.where(temp_acorr[0] == 1)])
                else:

                    TW_Dete_dict[ktw].append(0)
                    TW_Dete_dict[ktw].append(0)
                    TW_Dete_dict[ktw].append(0)
                # TODO find lag using CAF and PCAF
                plot_pacf(vtw, lags=7,title='Partial Autocorrelation:'+str(ktw))
                plot_acf(vtw, lags=7,title='Autocorrelation:'+str(ktw))
                plt.show()

        return TW_Dete_dict

    # Get the SD-Logs and Detected pattern and looks for periodic not active time step and remove them: generates new SD-Logs
    def Post_process_tw(self, SD_Log, TW_Dete_dict):

        # SD_Log = pd.read_csv("General2H_sdlog.csv")
        SD_Log = SD_Log.fillna(0)
        c = str(SD_Log.columns[0])[-2:]
        #temp_pattern = abs(TW_Dete_dict.get(c)[1])
        target_feature_values = SD_Log[SD_Log.columns[0]]
        new_df = SD_Log.loc[target_feature_values == 0]
        inactive_array = (new_df == 0).astype(int).sum(axis=1) / len(new_df.columns)
        my_list = list(inactive_array[inactive_array < 0.9].index)
        newnewdf = new_df.loc[my_list]
        ind = new_df.index
        Active_SD_Log = SD_Log.drop(SD_Log.index[ind])
        diff = list()
        for i in range(1, len(ind)):
            value = ind[i] - ind[i - 1]
            diff.append(value)
        # plt.bar([x for x in range(0,len(diff))],diff)
        plt.hist(diff)
        Active_SD_Log.to_csv(r"Active" + "_" + str(c) + "_sdlog.csv", index=False)
        return Active_SD_Log

    # Train different ARIMA Model for each time window and predict and calculate the error.
    def Detect_best_user_tw(self, TW_Dete_dict, Overall_dict):

        error_delta_dict = {}
        d = {}
        for k, v in Overall_dict.get("Arrival rate").items():
            v= v.values
            v = list(filter(lambda a: a != 0, v))
            if len(v) > 1:
                if (TW_Dete_dict.get(k)[1])!=0:
                    best_lag = abs(TW_Dete_dict.get(k)[1])
                    diff = list()
                    for i in range(1, len(v)):
                        value = v[i] - v[i - 1]
                        diff.append(value)
                    if len(diff) > 5:
                        if 0 < best_lag and best_lag < len(diff) // 5 and best_lag < 5:
                            arima_model = ARIMA(v, order=(best_lag, 0, 1))
                            model_fit = arima_model.fit(start_ar_lags=best_lag)
                            t = model_fit.predict()
                            error = np.mean(abs(v - t) / v) * 100
                            error_delta_dict[str(k)] = error

                        else:

                            arima_model = ARIMA(v, order=(1, 0, 1))
                            model_fit = arima_model.fit(start_ar_lags=2)
                            t = model_fit.predict()
                            error = np.mean(abs(v - t) / (v)) * 100
                            error_delta_dict[str(k)] = error
                    else:
                        error_delta_dict[str(k)] = -0.1

        lists = sorted(error_delta_dict.items())
        if len(lists) !=0:# sorted by key, return a list of tuples
            x, y = zip(*lists)
            plt.figure()
            plt.plot(x, y, '--bo')
            plt.title("Error of Different Time Delta")
            plt.xlabel("Time Delta")
            plt.ylabel("PAME")
            # plt.savefig('/UserTWError.png', dpi=100)
            plt.show()
        return
