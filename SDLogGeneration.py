import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import csv
import re

class SDLogGeneration:
    # Gets an event log with a list of time windows with different sizes: returns different SD-Logs
    def TW_discovery_process_calculation_twlist(self, event_log, tw_lists, aspect):
        Overall_dict = {}
        Arrival_rate_dict = {}
        event_log['Complete Timestamp'] = pd.to_datetime(event_log['Complete Timestamp'], errors='coerce')
        event_log['Start Timestamp'] = pd.to_datetime(event_log['Start Timestamp'], errors='coerce')
        event_log['Activity Duration'] = abs(event_log['Complete Timestamp'] - event_log['Start Timestamp'])
        start_unit_log = min(event_log['Start Timestamp'])
        end_log = max(event_log['Complete Timestamp'])
        arr_temp_event_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        fin_temp_event_log = event_log.sort_values(['Start Timestamp'], ascending=True).groupby(['Case ID'])
        case_dur_temp_log = event_log.groupby(['Case ID'])
        Name_General_selected_variables_dict = []
        for tw_list in tw_lists:
            # todo Arrival Rate of Cases per Day, Week, Month
            y = int((re.findall(r'\d+', tw_list))[0])
            z = tw_list[-1]

            if z == "H" and (24 % y == 0) or z != 'H':

                startdiff = []
                startforeach = []
                for actname, actgroup in arr_temp_event_log:
                    startforeach.append(actgroup['Start Timestamp'].values[0])
                startforeach.sort()

                for i in range(len(startforeach) - 1):
                    startdiff.append((startforeach[i + 1] - startforeach[i]))

                count_sta = Counter(startforeach)
                df_startforeach = pd.Series(count_sta).to_frame()
                Hourly = df_startforeach.resample(str(tw_list)).sum()
                Hourly.columns = ['hourly']

                # todo Finish Rate of Cases per Day, Week, Month
                enddiff = []
                endforeach = []
                for actname, actgroup in fin_temp_event_log:
                    endforeach.append(actgroup['Complete Timestamp'].values[-1])
                endforeach.sort()

                for i in range(len(endforeach) - 1):
                    enddiff.append((endforeach[i + 1] - endforeach[i]))

                enddiff = pd.to_timedelta(enddiff).seconds / 3600
                counter_intervals = Counter(enddiff)
                ecount_sta = Counter(endforeach)
                edf_startforeach = pd.Series(ecount_sta).to_frame()
                eHourly = edf_startforeach.resample(str(tw_list)).sum()

                eHourly.columns = ['ehourly']

                # Todo Compute the number of in process cases (MAX Capacity)
                Hourly_df = pd.concat([Hourly, eHourly], axis=1, join='outer')
                Hourly_df.fillna(0, inplace=True)

                h_case_in_process = (Hourly_df['hourly'] - Hourly_df['ehourly'])
                temp_list_inproc = h_case_in_process.tolist()
                for i in range(len(temp_list_inproc)):
                    if i == 0:
                        temp_list_inproc[i] = h_case_in_process.tolist()[i]
                    else:
                        temp_list_inproc[i] = temp_list_inproc[i] + temp_list_inproc[i - 1]

                max_h_case_in_process = max(h_case_in_process)

            # TODO Process whole active Time and Process active time per W,D,M
            try:
                process_active_time = event_log['Activity Duration'].sum()
            except ValueError as err:
                print(err)

            temp_active_time = event_log[['Start Timestamp', 'Activity Duration']]
            temp_active_time['Start Timestamp'] = pd.to_datetime(temp_active_time['Start Timestamp'])
            sort_start_duration = temp_active_time.sort_values('Start Timestamp')
            sort_start_duration.set_index('Start Timestamp', inplace=True)
            process_H_active_time_df = sort_start_duration.resample(str(tw_list)).sum()
            process_H_active_time = process_H_active_time_df['Activity Duration'].values
            process_H_active_time = pd.to_timedelta(process_H_active_time).total_seconds() / 3600
            # TODO List of time in process per case and real time service (Case Durations)
            case_duration_list = []
            case_real_duration_list = []
            case_real_duration_list_waiting = []
            case_duration_list_waiting = []
            case_duration_dict = {}
            case_real_duration_dict = {}
            for dcase, dgroup in case_dur_temp_log:
                case_real_duration_list_waiting.append(np.sum(dgroup['Activity Duration']))
                case_duration_list_waiting.append(
                    np.max(dgroup['Complete Timestamp']) - np.min(dgroup['Start Timestamp']))
                case_duration_dict[np.min(dgroup['Start Timestamp'])] = pd.to_timedelta(
                    np.max(dgroup['Complete Timestamp']) - np.min(dgroup['Start Timestamp'])).total_seconds() / 3600
                case_real_duration_dict[np.min(dgroup['Start Timestamp'])] = pd.to_timedelta(
                    np.sum(dgroup['Activity Duration'])).total_seconds() / 3600

            # case_duration_dict = {k: v for k, v in case_duration_dict.items() if v>0.5}
            # case_duration_dict = {k: v for k, v in case_duration_dict.items() if np.abs(v - np.mean(list(case_duration_dict.values())))<= (np.std(list(case_duration_dict.values())))}

            # case_real_duration_dict = {k: v for k, v in case_real_duration_dict.items() if v > 0.5}
            # case_real_duration_dict = {k: v for k, v in case_real_duration_dict.items() if np.abs(v - np.mean(list(case_real_duration_dict.values()))) <=(np.std(list(case_real_duration_dict.values())))}

            case_duration_list = list(case_duration_dict.values())
            case_real_duration_list = list(case_real_duration_dict.values())

            case_duration_df = pd.DataFrame(case_duration_dict.items(), columns=['Start Timestamp', 'Case Duration'])
            case_duration_df = case_duration_df.sort_values('Start Timestamp')
            case_duration_df.set_index('Start Timestamp', inplace=True)
            case_duration_H_df = case_duration_df.resample(str(tw_list)).sum()
            case_duration_H_df.insert(0, 'Avg Case Duration', case_duration_df.resample(str(tw_list)).mean())

            case_real_duration_df = pd.DataFrame(case_real_duration_dict.items(),
                                                 columns=['Start Timestamp', 'Case Duration'])
            case_real_duration_df = case_real_duration_df.sort_values('Start Timestamp')
            case_real_duration_df.set_index('Start Timestamp', inplace=True)
            case_real_duration_H_df = case_real_duration_df.resample(str(tw_list)).sum()
            case_real_duration_H_df.insert(0, 'Avg Case Duration', case_real_duration_df.resample(str(tw_list)).mean())
            case_real_duration_H_df.fillna(0)

            case_duration_list_waiting = pd.to_timedelta(case_duration_list_waiting).total_seconds() / 3600
            case_real_duration_list_waiting = pd.to_timedelta(case_real_duration_list_waiting).total_seconds() / 3600
            waiting_time = np.array(case_duration_list_waiting) - np.array(case_real_duration_list_waiting)

            case_duration_coutner = Counter(case_duration_list)
            case_real_duration_coutner = Counter(case_real_duration_list)
            avg_case_duration = np.mean(case_duration_list)
            avg_case_real_duration = np.mean(case_real_duration_list)

            # TODO Number of Resources and unique resources per
            temp_event_log_start_index = event_log.set_index('Start Timestamp')
            temp_group_h = temp_event_log_start_index.groupby(pd.Grouper(freq=str(tw_list)))

            num_unique_resource_h = (temp_group_h['Resource'].nunique()).values

            """
            finsihed_week =[]
            for sk in sklist:
                finish_counter = 0
                tem_com = temp_group_w.get(sk)
                for i in tem_com['Complete Timestamp']:
                    if sk>=i:
                        finish_counter += 1
                finsihed_week.append(finish_counter)
                """

            # TODO Number of Resources and unique resources per case
            uniqe_resource_list_per_case = []
            resource_list_per_case = []
            for rc, rgroup in case_dur_temp_log:
                resource_per_case = rgroup['Resource']
                unique_resource_per_case = np.unique(resource_per_case)
                uniqe_resource_list_per_case.append(len(unique_resource_per_case))
                resource_list_per_case.append(len(resource_per_case))

            # TODO Create Overall Dict

            Arrival_rate_dict[str(tw_list)] = Hourly['hourly'].values.tolist()

            Overall_dict["Arrival rate"] = Arrival_rate_dict
            Name_General_selected_variables_dict.append(str(aspect) + "_" + str(tw_list) + "_sdlog.csv")
            General_selected_variables_dict = {"Arrival rate" + str(tw_list): Hourly['hourly'].values.tolist(),
                                               "Finish rate" + str(tw_list): (eHourly['ehourly'].values).tolist(),
                                               "Num of unique resource" + str(tw_list): num_unique_resource_h.tolist(),
                                               "Process active time" + str(tw_list): case_duration_H_df[
                                                   'Case Duration'].tolist(),
                                               "Service time per case" + str(tw_list): case_real_duration_H_df[
                                                   'Avg Case Duration'].tolist(),
                                               "Time in process per case" + str(tw_list): case_duration_H_df[
                                                   'Avg Case Duration'].tolist(),
                                               "Waiting time in process per case" + str(tw_list): pd.array(
                                                   case_duration_H_df['Avg Case Duration'].tolist()) - pd.array(
                                                   case_real_duration_H_df['Avg Case Duration'].tolist()),
                                               "Num in process case" + str(tw_list): temp_list_inproc,
                                               }
            with open(str(aspect) + "_" + str(tw_list) + "_sdlog.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerow(General_selected_variables_dict.keys())
                x = zip(*General_selected_variables_dict.values())
                xx = zip(*x)
                xxx = zip(*xx)
                writer.writerows(xxx)


        return Overall_dict, Name_General_selected_variables_dict
