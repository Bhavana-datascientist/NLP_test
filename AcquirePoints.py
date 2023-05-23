import pandas as pd
import metric_computation
import protobuf_packaging
import utils
import pdb
from config_dict import config
import numpy as np

closest = lambda list_values, value: np.argmin(np.abs(np.array(list_values) - value))


def post_acquire_closest(list_values, value):
    cls_min = np.array(list_values) - value
    return np.argmin(np.abs(np.array(cls_min)-0))


class AcquirePoints:
    def __init__(self, storage, proc_pool, acquire_timestamp, conn_sender):
        self.storage_obj = storage
        self.proc_pool = proc_pool
        self.acquire_timestamp = acquire_timestamp
        self.computed_metric_df = pd.DataFrame()
        self.conn_sender = conn_sender

    def find_the_nearest(self, matched_to, to_match):
        return list([post_acquire_closest(matched_to, i) for i in to_match]) # list(filter(None, list(set([
        # post_acquire_closest(matched_to, i) for i in to_match]))))
        # return list(set([closest(matched_to, i) for i in to_match]))

    def beat_selection(self):

        storage_keys = list(map(float, list(self.storage_obj.dict_beats.keys())))
        matched_beat_list = self.find_the_nearest(storage_keys, self.acquire_timestamp)
        if len(matched_beat_list) == 0:
            utils.record_statement("#########No beats detected#############")
            return []

        matched_beat_list = [self.storage_obj.dict_beats[str(storage_keys[i])] for i in matched_beat_list]

        matched_beat_list_threshold = []
        for i in range(len(matched_beat_list)):
            if (abs(self.acquire_timestamp[i] - matched_beat_list[i].end_time)) > config["Threshold_beat_selection"]:
                utils.record_statement("matching beat was not selected")
                protobuf_packaging.rhythm_change_detected(self.conn_sender,message="No nearest matching beat",code=1005)
                utils.record_statement("time difference for beat and acquire time {}".format(
                    self.acquire_timestamp[i] - matched_beat_list[i].end_time))
                utils.record_statement(self.storage_obj.dict_beats)
                utils.record_statement(str(self.acquire_timestamp[i])+" is the acquire and beat mapped is "+str(matched_beat_list[i].end_time))
            else:
                utils.record_statement("beat has been matched. Acquire timestmap : {} and beat matched timestamp : {}"
                                       .format(self.acquire_timestamp[i],  matched_beat_list[i].end_time))
                matched_beat_list_threshold.append(matched_beat_list[i])

        return matched_beat_list_threshold

    def lr_df_creation(self, previous_pt_number, matched_beats):

        utils.record_statement("beat_selection - starts")
        df_columns = config["lr_df_columns"]
        df = pd.DataFrame(columns=df_columns)
        metric_df = pd.DataFrame()

        list_of_rows = []
        count = 0

        for matched_beat in matched_beats:

            coordinate_df = self.storage_obj.coordinate_cummulative_df
            if coordinate_df.shape[0] == 0:
                continue
            beat_coordinates = coordinate_df[(coordinate_df.timestamp >= matched_beat.start_time) & (coordinate_df.timestamp <= matched_beat.end_time)]
            xyz_resolved_dict = self.xyz_resolution(beat_coordinates)

            if len(xyz_resolved_dict) == 0:
                utils.record_statement("No NDI data...")
                utils.record_statement("beat_selection - ends")
                continue

            for node, egm in matched_beat.egm.items():
                if config["only_bipolar"]:
                    node_bipolar = node
                else:
                    node_bipolar = config["unipolar_bipolar_mapping"][node]

                mapping_point = xyz_resolved_dict[node]

                list_of_rows.append([matched_beat.start_time, matched_beat.end_time, mapping_point[0], mapping_point[1],
                                     mapping_point[2],mapping_point[3],mapping_point[4],mapping_point[5], matched_beat.ref_ecg, egm["unipolar"], egm["bipolar"], 1, 0,
                                     previous_pt_number, 1, 1, 0, 0, 1, node, node_bipolar, 0, matched_beat.ecg,
                                     matched_beat.ndi_start_time, matched_beat.ndi_end_time,
                                     matched_beat.end_time])
            count += 1
            previous_pt_number += 1

            metric_df = pd.concat([metric_df, matched_beat.metrics], ignore_index=True)

        df = df.append(pd.DataFrame(list_of_rows, columns=df_columns), ignore_index=True)
        df = pd.concat([df, metric_df], axis=1)

        df = df[df["rov_trace"].isin(config["selected_nodes"])].reset_index(drop=True)
        if df.shape[0] == 0:
            return pd.DataFrame(), previous_pt_number
        if config["only_bipolar"]:
            df.drop(columns=['egm', 'qrs_onset', 'iqrs', 'activation_time', 'minimum_derivative', 'peak2peak',
                             'fractionation', 'ipk', 'inadr', 'activation_index'], inplace=True)

        utils.record_statement("beat_selection - ends")

        if df.shape[0] > len(matched_beats):
            pdb.set_trace()

        return df, previous_pt_number

    def xyz_resolution(self, df):
        utils.record_statement("xyz_resolution - starts")

        xyz_resolved_dict = {}
        for node in df.node_name.unique():
            df_node = df[df["node_name"] == node].reset_index(drop=True)
            # df_node_mean = df_node.mean()

            df_node_mean = df_node.loc[df_node.index[-1], ["x", "y", "z","raw_x","raw_y","raw_z"]]
            # else:
            #     df_node_mean = df_node.loc[df_node.index[-1], ["x", "y", "z"]]
            #     df_node_mean["raw_x"], df_node_mean["raw_y"], df_node_mean["raw_z"] = [np.nan, np.nan, np.nan]

            xyz_resolved_dict[node] = df_node_mean[["x", "y", "z","raw_x","raw_y","raw_z"]].values
        utils.record_statement("xyz_resolution - ends")

        return xyz_resolved_dict

    def metric_computation(self, df):
        utils.record_statement("Metric computation - starts")

        measured_points_df = df[(df["mapped"] == 1.0) & (df["metric_computed"] == 0)]
        unmeasured_points_df = df[~df.index.isin(measured_points_df.index)]

        utils.record_statement("measured_points_df unmeasured_points_df formations...")

        mp = list(measured_points_df.index)

        if len(mp) > 0:
            selected_ecg = measured_points_df["ecg"].values
            selected_egm = measured_points_df["egm"].values
            print("# LR metrics computed : ", len(selected_ecg))
            utils.record_statement("waveform values taking out done...")

            # Selecting a particular beat
            utils.record_statement("select_single_beat done...")

            # for i in range(len(selected_ecg)):
            #     temp = metric_computation.computing_new_lux_metrics(selected_ecg[i], selected_egm[i])

            results = self.proc_pool.starmap(metric_computation.computing_new_lux_metrics,
                                             zip(selected_ecg, selected_egm))

            utils.record_statement("multiprocessing of compute_metrics_cpp done...")

            results = pd.DataFrame(results, index=measured_points_df.index)

            utils.record_statement("df formation done...")

            measured_points_df = pd.concat([measured_points_df, results], axis=1, ignore_index=False)
            measured_points_df["metric_computed"] = 1

            utils.record_statement("concatation done...")

            df = pd.concat([measured_points_df, unmeasured_points_df], axis=0)
            df.sort_index(inplace=True)

            utils.record_statement("concatation-2 done...")

            utils.record_statement("Metric computation - ends")

        return df

    def send_acquire_point_protobuf(self):
        return None

    def respiration_gate(self, df):
        df["respiration_gate"] = 1
        return df

    @staticmethod
    def update_tag_index(df, return_dict):
        df.at[df.index, ["tag_index"]] = list(range(return_dict["tag_index"], return_dict["tag_index"] + len(df.index)))
        return_dict["tag_index"] += len(df.index)
        return df, return_dict

    def complete_process(self, return_dict):

        if len(self.storage_obj.dict_beats) == 0:
            utils.record_statement("No beats are present to find the acquired beat...", forced="print")
            utils.record_statement("No beats are present to find the acquired beat...")
            return return_dict

        # beat_selection is called to find the nearest beat using the timestamp
        matched_beat_list = self.beat_selection()
        if len(matched_beat_list) > 0:
            # The data is converted into a dataframe and the updated PT number is returned.
            beat_selection_df, previous_pt_number = self.lr_df_creation(return_dict["NT_PT_NO"], matched_beat_list)
            return_dict["NT_PT_NO"] = previous_pt_number

            beat_selection_df, return_dict = self.update_tag_index(beat_selection_df, return_dict)

            # Check is the beat is inside the respiration window.
            self.computed_metric_df = self.respiration_gate(beat_selection_df)

            if self.computed_metric_df.shape[0] > 0:
                return_dict["lr_df_changed"] = True
            else:
                return_dict["lr_df_changed"] = False
                utils.record_statement("No matching beats....")
        # else:
        #     return_dict["lr_df_changed"] = False
        #     utils.record_statement("No matching beats.. with matched_beat_list == 0..")
        return return_dict
