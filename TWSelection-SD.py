import pandas as pd
from SDLogGeneration import SDLogGeneration
from TWSelection import TWSelection


# Sample step for the time window selection appraoch
if __name__=="__main__":
    tws = TWSelection()
    sdgen = SDLogGeneration()
    tw_result=""
    event_log= "event_log.csv"
    event_log = pd.read_csv(event_log)
    tw_lists = ['1H','7D','1W'] # sample time window lists = [8H,2D,1W,1M]
    tw_discovered = sdgen.TW_discovery_process_calculation_twlist(event_log=event_log, tw_lists=tw_lists, aspect="General")
    generated_SD_log = tw_discovered[1]
    overall_dict = tw_discovered[0]
    TW_Dete_dict = tws.Detect_pattern_tw(overall_dict, event_log)
    tw_best = max(TW_Dete_dict, key=TW_Dete_dict.get)
    tw_worse = min(TW_Dete_dict, key=TW_Dete_dict.get)
    for k, v in TW_Dete_dict.items():
        if abs(v[2]) > 0.5:
            tw_result = tw_result + ' ' + str(k) + ' is a strong pattern.'
        else:
            tw_result = tw_result + ' ' + str(k) + ' is not a strong pattern.'

    tw_result = tw_result + "\n The Strongest Pattern Discovered: " + str(
        abs(TW_Dete_dict.get(tw_best)[1])) + " " + str(tw_best) + "\n"
    # "\n The Weakest Pattern: "+str(abs(TW_Dete_dict.get(tw_worse)[1]))+' For ' +str(tw_worse)

    active_overall_dict = {}
    active_overall_dict["Arrival rate"] = {}
    for sdlog_name in generated_SD_log:
        sd_log = pd.read_csv(sdlog_name)
        name_sd_tw = sdlog_name.split("_")[1]
        overall_dict["Arrival rate"].get(name_sd_tw)
        Active_SD_Log = (tws.Post_process_tw(sd_log, TW_Dete_dict))
        active_overall_dict["Arrival rate"][name_sd_tw] = Active_SD_Log[Active_SD_Log.columns[0]]

        active_TW_Dete_dict = tws.Detect_pattern_tw(active_overall_dict, event_log)

    tws.Detect_best_user_tw(active_TW_Dete_dict, active_overall_dict)