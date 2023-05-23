import base64
import pdb
import pickle
import time
import traceback
import win32api
import data_subscriber
import protobuf_packaging
from config_dict import config
import pandas as pd
import logging
import warnings
import utils
import Storage
import Catheter
import Geometry
import Interpolation
from multiprocessing.pool import ThreadPool
import os
import logs_backup
import atexit
import Internal_functions

warnings.filterwarnings('ignore')

with warnings.catch_warnings():
    warnings.simplefilter("always")

logging.getLogger("stomp").setLevel(logging.WARNING)


def main():
    temp_dir = utils.get_tmp_dir()
    temp_logs_dir = os.path.join(temp_dir, "logs")
    temp_backup_dir = os.path.join(temp_dir, "logs_back_up")
    crash_backup_dir = os.path.join(temp_dir, "back_up")
    utils.create_folder(crash_backup_dir)
    utils.create_folder(temp_logs_dir)
    utils.create_folder(temp_backup_dir)

    def crash_back_up():
        utils.back_up(map_return_dict, crash_backup_dir, "acquire_config.pkl")
        logs_backup_obj.write_file("all_logs_backup.pkl")
        utils.back_up(map_return_dict, temp_logs_dir, "return_dict_{}.pkl".format(map_return_dict["current_map"]))

    def exception_handler(sig=None, func=None):
        crash_back_up()
        utils.copy_folder_tree(source=temp_logs_dir, dest=temp_backup_dir)

    win32api.SetConsoleCtrlHandler(exception_handler, 1)
    current_map = config["acquire_return_dict"]["map_id"]
    map_return_dict = {"current_map": current_map, current_map: config["acquire_return_dict"].copy()}
    map_return_dict[current_map]["storage"] = Storage.StorageAcquire()

    logs_backup_obj = logs_backup.backup()

    acquire_logger = utils.setup_logger('acquisition', os.path.join(temp_logs_dir, 'app_acquire.log'),
                                        level=logging.INFO)
    # second file logger
    data_formatter = '%(message)s'
    super_logger = utils.setup_logger('second_logger', os.path.join(temp_logs_dir, 'acquire_point.log'), data_formatter,
                                      level=logging.INFO)

    acquire_logger.info("ACQUIRE_POINT EXECUTION STARTED ..")
    # create a dictionary for all config variables

    catheter_obj = Catheter.Catheter()

    # Creating a multiprocessing pool. This is used every time a multiprocessing is called.
    proc_pool = ThreadPool(processes=utils.num_processes("all"))

    # Creating the connectors and listener to access the RabbitMQ
    acquire_logger.info("ActiveMQ subscriber data started")

    conn_sender = data_subscriber.sender_connection()
    conn_sender_localhost = data_subscriber.sender_connection(config["local_host"], config["local_port"],
                                                              config["local_host_username"],
                                                              config["local_host_password"])

    beats_listener, beats_receiver = data_subscriber.receiver_connection(config["local_host"], config["local_port"],
                                                                         config["local_host_username"],
                                                                         config["local_host_password"])
    acquire_listener, acquire_receiver = data_subscriber.receiver_connection()
    xyz_listener, xyz_receiver = data_subscriber.receiver_connection(vhost="datapub")

    conn_listener_dict = {"beats": {"listener": beats_listener, "conn_receiver": beats_receiver},
                          "acquired_time": {"listener": acquire_listener, "conn_receiver": acquire_receiver},
                          "coordinates": {"listener": xyz_listener, "conn_receiver": xyz_receiver}}
    acquire_logger.info("estalish_connection done...")

    data_subscriber.connect_and_subscribe(conn_listener_dict["beats"]["conn_receiver"],
                                          config["beats_queue"],
                                          headers={"durable": True, "auto-delete": config["auto_delete"]})
    data_subscriber.connect_and_subscribe(conn_listener_dict["acquired_time"]["conn_receiver"],
                                          config["acquire_time_stamp"],
                                          headers={"durable": True, "auto-delete": config["QT_auto_delete"]})
    data_subscriber.connect_and_subscribe(conn_listener_dict["coordinates"]["conn_receiver"],
                                          config["acquire_coordinates"],
                                          headers={"durable": True, "auto-delete": config["auto_delete"]})
    if config["simulation_as_realtime"]:
        acquire_config_path = os.path.join(temp_logs_dir, "return_dict_la2.pkl")
        if os.path.exists(acquire_config_path):
            map_return_dict = pickle.load(open(acquire_config_path, "rb"))
            print("loaded the pkl file...")
            current_map = map_return_dict['current_map']
            map_return_dict[current_map]["ndi_df_changed"] = True
            map_return_dict[current_map]["lr_df_changed"] = True
            map_return_dict[current_map]["storage"].previous_acquired_points = []
            map_return_dict[current_map]["last_sinus_coordinate"] = {}
            map_return_dict[current_map]["annulus_status"] = False
            map_return_dict[current_map]["annulus_data"] = []
    if config["backup_files"] and not config["simulation_as_realtime"]:
        acquire_config_path = os.path.join(crash_backup_dir, "acquire_config.pkl")
        if os.path.exists(acquire_config_path):
            try:
                map_return_dict = pickle.load(open(acquire_config_path, "rb"))
                current_map = map_return_dict['current_map']
                map_return_dict[current_map]["ndi_df_changed"] = True
                map_return_dict[current_map]["lr_df_changed"] = True
                map_return_dict[current_map]["storage"].cummulative_lr_df["wandering"] = False
                map_return_dict[current_map]["storage"].cummulative_lr_df["auto_acquire"] = False
                map_return_dict[current_map]["last_sinus_coordinate"] = {}
                map_return_dict[current_map]["annulus_status"] = False
                map_return_dict[current_map]["annulus_data"] = []
                os.remove(acquire_config_path)
            except EOFError:
                os.remove(acquire_config_path)
    acquire_timestamp = []
    acquire_received_time = None
    ndi_frequency_constant = config["ndi_frequency_for_geometry"] * config["N_nodes"]
    auto_map_dict = {"timestamp": time.time() * 1000, "auto_acquire": False, "auto_geo": False, "manual_acquire": False}
    rule_engine_patient_data = {}
    isochone_computation = False
    last_single_beat = False
    return_dict_time = time.time()*1000
    # Looping forever
    while True:
        try:
            time.sleep(0.1)
            atexit.register(exception_handler)
            if (time.time()*1000 - return_dict_time) > config["back_up_writing_interval"]:
                acquire_logger.info("writing the return dict starts")
                crash_back_up()
                return_dict_time = time.time()*1000
                acquire_logger.info("writing the return dict ends")
            # Subscribing the data from the queue

            beats_messages = conn_listener_dict["beats"]["listener"].message.copy()
            time_stamp_message = conn_listener_dict["acquired_time"]["listener"].message.copy()
            coordinate_messages = conn_listener_dict["coordinates"]["listener"].message.copy()

            auto_map, delete_points_dict, undo_delete_points, ablation_req, update_tag, update_metrics, \
            waveform_template, new_map_req, delete_faces, auto_map_status, selected_map, end_map_status, \
            logs_backup_obj, geo_config, final_geo, ref_template_req, plane_cutter,plane_cutter_undo = protobuf_packaging.client_request_handler(
                time_stamp_message,
                logs_backup_obj)
            auto_map_dict.update(auto_map)
            # rule_engine_patient_data.update(patient_info)
            if not auto_map_dict["auto_geo"] and not auto_map_dict["auto_acquire"] and isochone_computation:
                if hasattr(map_return_dict[current_map]["storage"], "interpolated_geo_df"):
                    # protobuf_packaging.connect_and_send(
                    #     map_return_dict[current_map]["storage"].interpolated_geo_df.copy(), conn_sender,
                    #     map_return_dict[current_map], False)
                    isochone_computation = False
                    protobuf_packaging.dataframe_to_protobuf(
                        map_return_dict[current_map]["storage"].interpolated_geo_df.copy(), conn_sender,
                        map_return_dict[current_map], False)
                    print("propogation data sent to model :{}".format(current_map))
            if end_map_status:

                logs_backup_obj.write_file("all_logs_backup.pkl")

                utils.back_up(map_return_dict, temp_logs_dir,
                              "return_dict_{}.pkl".format(map_return_dict["current_map"]))
                utils.copy_folder_tree(source=temp_logs_dir, dest=temp_backup_dir)

                auto_map_dict = {"timestamp": time.time() * 1000, "auto_acquire": False, "auto_geo": False,
                                 "manual_acquire": False}
                rule_engine_patient_data = {}

                current_map = config["acquire_return_dict"]["map_id"]
                map_return_dict = {"current_map": current_map, current_map: config["acquire_return_dict"].copy()}

            if new_map_req:
                current_map = new_map_req["map_id"]  # new_map_req["waveform_template"]["map_id"]
                if not (current_map in map_return_dict.keys()):

                    logs_backup_obj.write_file("all_logs_backup.pkl")

                    utils.back_up(map_return_dict, temp_logs_dir,
                                  "return_dict_{}.pkl".format(map_return_dict["current_map"]))
                    map_return_dict = dict()
                    map_return_dict[current_map] = config["acquire_return_dict"].copy()
                    map_return_dict[current_map]["storage"] = Storage.StorageAcquire()
                    map_return_dict[current_map]["volatility"] = []
                    map_return_dict[current_map]["ablation_id"] = 0
                    map_return_dict[current_map]["deleted_points"] = []
                    map_return_dict[current_map]["rule_engine_patient_data"] = {}
                    map_return_dict[current_map]["last_sinus_coordinate"] = {}
                    map_return_dict[current_map]["annulus_status"] = False
                    map_return_dict[current_map]["annulus_data"] = []

                map_return_dict["current_map"] = current_map
                map_return_dict[current_map]["map_id"] = current_map
                map_return_dict[current_map]["map"] = new_map_req
                map_return_dict[current_map]["final_geo"] = False
                auto_map_dict = {"timestamp": time.time() * 1000, "auto_acquire": False, "auto_geo": False,
                                 "manual_acquire": False}

            if selected_map:
                print("switched the map to {}:".format(selected_map))

                logs_backup_obj.write_file("all_logs_backup.pkl")
                utils.back_up(map_return_dict, temp_logs_dir,
                              "return_dict_{}.pkl".format(map_return_dict["current_map"]))

                acquire_config_path = os.path.join(temp_logs_dir, "return_dict_{}.pkl".format(selected_map))
                if os.path.exists(acquire_config_path) and current_map != selected_map:
                    map_return_dict = pickle.load(open(acquire_config_path, "rb"))
                    current_map = map_return_dict['current_map']
                    map_return_dict[current_map]["map_id"] = current_map
                    map_return_dict[current_map]["deleted_points"] = []

                    auto_map_dict = {"timestamp": time.time() * 1000, "auto_acquire": False, "auto_geo": False,
                                     "manual_acquire": False}
                    rule_engine_patient_data = map_return_dict[current_map]["rule_engine_patient_data"]
                    template_data = \
                        Internal_functions.template_detection(map_return_dict[current_map]["map"],
                                                              map_return_dict)
                    protobuf_packaging.send_waveform_template(template_data, map_return_dict[current_map]["map"],
                                                              conn_sender_localhost, map_return_dict)
                    # protobuf_packaging.send_qrs_reference(map_return_dict[current_map]["map"], conn_sender)
                else:
                    print("{} file doesn't exist".format(acquire_config_path))

            # Deleting the messages which has been subscribed from the queue
            del conn_listener_dict["beats"]["listener"].message[0:len(beats_messages)]
            del conn_listener_dict["acquired_time"]["listener"].message[0:len(time_stamp_message)]
            del conn_listener_dict["coordinates"]["listener"].message[0:len(coordinate_messages)]
            del time_stamp_message[:len(time_stamp_message)]

            if ref_template_req:
                protobuf_packaging.send_rhy_template_reference(conn_sender,ref_template_req)
                print("waveform template reference sent")
            # continue and wait was new map get created
            if not current_map:
                continue
            map_return_dict[current_map]["auto_map_dict"] = auto_map_dict
            map_return_dict[current_map]["rule_engine_patient_data"] = rule_engine_patient_data

            acquire_logger.info("auto_map status is : {}".format(auto_map_dict))
            auto_map_received_at = time.time() * 1000

            # if waveform_template:
            #     template_data = \
            #         Internal_functions.template_detection(waveform_template,
            #                                               map_return_dict)
            #     protobuf_packaging.send_waveform_template(template_data, conn_sender_localhost, map_return_dict)
            #
            #     print("template sent for beat detection")

            if new_map_req:
                # protobuf_packaging.send_qrs_reference(new_map_req, conn_sender)

                template_data = \
                    Internal_functions.template_detection(new_map_req,
                                                          map_return_dict)
                protobuf_packaging.send_waveform_template(template_data, new_map_req, conn_sender_localhost, map_return_dict)
                acquire_logger.info("new map created request:{}".format(new_map_req))
                print("template sent for beat detection")

            if len(coordinate_messages) > 0 and map_return_dict["current_map"] is not None:
                acquire_logger.info("converting NDI into a dataframe starts..")
                coordinate_messages_json_loads = utils.data_preparation_coordinates(coordinate_messages)
                acquire_logger.info("coordinate_message_received : " + str(coordinate_messages_json_loads.shape[0]))

                if coordinate_messages_json_loads.shape[0] > 0:
                    respiration_corrected = coordinate_messages_json_loads
                    if config["respiration_correction"]:
                        respiration_corrected = catheter_obj.respiration_correction_method(
                            coordinate_messages_json_loads,
                            map_return_dict[current_map])
                    acquire_logger.info("convert_tip_to_nodes starts..")
                    respiration_corrected['raw_x'], respiration_corrected['raw_y'], respiration_corrected['raw_z'] = \
                        coordinate_messages_json_loads['Tx'], coordinate_messages_json_loads['Ty'], \
                        coordinate_messages_json_loads['Tz']
                    coordinate_messages_df = catheter_obj.convert_tip_to_nodes(proc_pool, respiration_corrected,
                                                                               map_return_dict[current_map])

                    coordinate_messages_df.index = [
                        map_return_dict[current_map]["storage"].coordinate_cummulative_df.shape[0] + i for i in
                        range(coordinate_messages_df.shape[0])]

                    acquire_logger.info("convert_tip_to_nodes ends..")

                    acquire_logger.info("accumulating the data received starts..")
                    acquire_logger.info("coordinates_cumulative_data shape {}".format(
                        map_return_dict[current_map]["storage"].coordinate_cummulative_df.shape[0]))
                    map_return_dict[current_map]["storage"].append_coordinate_cummulative_df(
                        coordinate_messages_df.copy())
                    acquire_logger.info("accumulating the data received ends..")

                    if config["post_procedure"]:
                        present_time = coordinate_messages_df.timestamp.values.tolist()[-1]
                    else:
                        present_time = time.time() * 1000

                    acquire_logger.info("# storage_obj.coordinate_cummulative_df : " +
                                        str(map_return_dict[current_map]["storage"].coordinate_cummulative_df.shape[
                                                0]))

            if not map_return_dict[current_map]["storage"].coordinate_cummulative_df.empty:
                map_return_dict[current_map]["storage"].purge_coordinate_cummulative_df(
                    max(map_return_dict[current_map]["storage"].coordinate_cummulative_df.timestamp) - 60000)

            if not map_return_dict[current_map]["storage"].manual_beat_df.empty:
                map_return_dict[current_map]["storage"].purge_manual_beat_df(
                    max(map_return_dict[current_map]["storage"].manual_beat_df.end_time) - 60000)
                acquire_logger.info("# manual_beat_df : " +
                                    str(map_return_dict[current_map]["storage"].manual_beat_df.shape[0]))

            if len(beats_messages) > 0 and map_return_dict[current_map]["storage"].coordinate_cummulative_df.shape[
                0] > 0:

                beats_decoded = [base64.b64decode(i) for i in beats_messages]
                beats_decoded_loaded = [pickle.loads(i) for i in beats_decoded]

                acquire_logger.info("updating the beats in storage starts..")
                min_time_in_coordinates = min(
                    map_return_dict[current_map]["storage"].coordinate_cummulative_df.timestamp.to_list())

                coordinate_cummulative_df = pd.DataFrame()
                for beat in beats_decoded_loaded:
                    for key, value in beat.items():
                        pickle_loaded = pickle.loads(value)
                        beat_df, last_sinus_coordinate = utils.beat_coordinates(
                            map_return_dict[current_map]["storage"].coordinate_cummulative_df, pickle_loaded,
                            map_return_dict[current_map]["last_sinus_coordinate"])
                        map_return_dict[current_map]["last_sinus_coordinate"] = last_sinus_coordinate
                        coordinate_cummulative_df = pd.concat(
                            [coordinate_cummulative_df.reset_index(drop=True),
                             beat_df.reset_index(drop=True)], axis=0)

                if not coordinate_cummulative_df.empty:
                    if auto_map_dict["auto_acquire"]:
                        if len(map_return_dict[current_map]["storage"].cumulative_beat_df) > 0:
                            start_index = int(max(map_return_dict[current_map]["storage"].cumulative_beat_df.tag_index))
                        else:
                            start_index = 0

                        coordinate_cummulative_df["tag_index"] = list(
                            range(start_index + 1, start_index + 1 + len(coordinate_cummulative_df)))

                        coordinate_cummulative_df_geo = coordinate_cummulative_df[
                            coordinate_cummulative_df["geo_beat"] == True]

                        acquire_logger.info(
                            "length of coordinate_cummulative_df_geo: " + str(
                                len(coordinate_cummulative_df_geo)))

                        # Appending the data to the cumulative beat df
                        map_return_dict[current_map]["storage"].cumulative_beat_df = pd.concat(
                            [map_return_dict[current_map]["storage"].cumulative_beat_df,
                             coordinate_cummulative_df_geo]).reset_index(drop=True)

                        coordinate_cummulative_df["cumulative_beat_tag_index"] = \
                            coordinate_cummulative_df["tag_index"]
                        if config["distance_thresholding_special_points"]:
                            # 2mm thresholding for acquire points
                            coordinate_cummulative_df = \
                                utils.find_coordinate_greatest(
                                    map_return_dict[current_map]["storage"].cummulative_lr_df.copy(),
                                    coordinate_cummulative_df, config["spacial_thresholding_auto_acquire"])

                        if len(coordinate_cummulative_df) > 0:
                            coordinate_cummulative_df = coordinate_cummulative_df[
                                coordinate_cummulative_df["node_name"].isin(
                                    config["selected_nodes_for_electrical"])]

                            if coordinate_cummulative_df.shape[0] > 0:
                                if config["no_bad_beat_for_auto_electrical"]:
                                    coordinate_cummulative_df = coordinate_cummulative_df[
                                        coordinate_cummulative_df["bad_beat"] == False].reset_index(drop=True)

                                acquire_logger.info(
                                    "length of df before sub sampling: " + str(len(coordinate_cummulative_df)))

                                if config["sub_sampling_electrical"]:
                                    if last_single_beat and len(coordinate_cummulative_df) == 1:
                                        coordinate_cummulative_df = pd.DataFrame()
                                        last_single_beat = False
                                    elif not last_single_beat and len(coordinate_cummulative_df) == 1:
                                        last_single_beat = True

                                coordinate_cummulative_df = coordinate_cummulative_df.iloc[
                                                            ::config["beats_frequency_for_geometry"], :]

                                acquire_logger.info(
                                    "length of df after sub sampling: " + str(len(coordinate_cummulative_df)))
                                coordinate_cummulative_df_electrical = coordinate_cummulative_df[
                                    coordinate_cummulative_df["electric_beat"] == True]
                                acquire_logger.info(
                                    "length of df after considering only electrical beats: " + str(
                                        len(coordinate_cummulative_df_electrical)))

                                coordinate_cummulative_df_electrical["auto_acquire"] = True
                                map_return_dict[current_map]["storage"].cummulative_lr_df = \
                                    pd.concat([map_return_dict[current_map]["storage"].cummulative_lr_df,
                                               coordinate_cummulative_df_electrical]).reset_index(drop=True)

                                # TODO: Need to confirm if this is needed
                                coordinate_cummulative_df_electrical_not_in_geo = \
                                    coordinate_cummulative_df_electrical[(coordinate_cummulative_df_electrical["electric_beat"] == True) & (coordinate_cummulative_df_electrical["geo_beat"] == False)]
                                map_return_dict[current_map]["storage"].cumulative_beat_df = pd.concat(
                                    [map_return_dict[current_map]["storage"].cumulative_beat_df,
                                     coordinate_cummulative_df_electrical_not_in_geo]).reset_index(drop=True)

                                map_return_dict[current_map]["storage"].cummulative_lr_df["tag_index"] = \
                                    map_return_dict[current_map]["storage"].cummulative_lr_df["tag_index"].index

                    elif auto_map_dict["auto_geo"]:
                        coordinate_cummulative_df_geo = coordinate_cummulative_df[
                            coordinate_cummulative_df["geo_beat"] == True]

                        if len(map_return_dict[current_map]["storage"].cumulative_beat_df) > 0:
                            start_index = int(max(map_return_dict[current_map]["storage"].cumulative_beat_df.tag_index))
                        else:
                            start_index = 0

                        coordinate_cummulative_df_geo["tag_index"] = list(
                            range(start_index + 1, start_index + 1 + len(coordinate_cummulative_df_geo)))

                        map_return_dict[current_map]["storage"].cumulative_beat_df = pd.concat(
                            [map_return_dict[current_map]["storage"].cumulative_beat_df,
                             coordinate_cummulative_df_geo]).reset_index(drop=True)

                        # map_return_dict[current_map]["storage"].cumulative_beat_df["tag_index"] = \
                        #     map_return_dict[current_map]["storage"].cumulative_beat_df.index

                    else:
                        map_return_dict[current_map]["storage"].manual_beat_df = pd.concat(
                            [map_return_dict[current_map]["storage"].manual_beat_df,
                             coordinate_cummulative_df]).reset_index(drop=True)

                        map_return_dict[current_map]["storage"].manual_beat_df["tag_index"] = -1

                acquire_logger.info("# beats subscribed : " + str(len(beats_decoded_loaded)))
                acquire_logger.info("updating the beats in storage ends..")
                acquire_logger.info("Difference on receiving the beat : " + str(time.time() * 1000 - float(key)))

            if len(delete_points_dict) > 0:
                map_return_dict[current_map]["storage"] = protobuf_packaging.surface_delete_points(
                    map_return_dict[current_map], delete_points_dict)
                map_return_dict[current_map]["lr_df_changed"] = True
                map_return_dict[current_map]["deleted_points"].append(delete_points_dict)
                acquire_logger.info("deleted surface point cloud")

            if len(rule_engine_patient_data) > 0:
                utilized_beats = map_return_dict[current_map]["storage"].cummulative_lr_df.loc[
                                 map_return_dict[current_map]["storage"].cummulative_lr_df.utilized == 1, :]
                rule_engine_patient_data["points"] = utilized_beats.shape[0]
                if "minimum_derivative_bipolar" in utilized_beats.columns:
                    rule_engine_patient_data["points_threshold"] = \
                        utilized_beats[
                            utilized_beats.minimum_derivative_bipolar >= config["rule_engine_min_dvdt"]].shape[0]
                    rule_engine_patient_data["points_threshold"] = (rule_engine_patient_data["points_threshold"] /
                                                                    rule_engine_patient_data["points"]) * 100
                elif "minimum_derivative" in utilized_beats.columns:
                    rule_engine_patient_data["points_threshold"] = utilized_beats[
                        utilized_beats.minimum_derivative >= config["rule_engine_min_dvdt"]].shape[0]
                    rule_engine_patient_data["points_threshold"] = (rule_engine_patient_data["points_threshold"] /
                                                                    rule_engine_patient_data["points"]) * 100
                else:
                    rule_engine_patient_data["points_threshold"] = 0

                protobuf_packaging.ask_experts_rule(rule_engine_patient_data)

            if len(undo_delete_points) > 0:
                if len(map_return_dict[current_map]["deleted_points"]) == 0:
                    print("there is no points in stack to do UNDO operation")
                    map_return_dict[current_map]["lr_df_changed"] = False
                else:
                    map_return_dict[current_map] = protobuf_packaging.undo_deleted_points(map_return_dict[current_map])
                    map_return_dict[current_map]["lr_df_changed"] = True

            if plane_cutter_undo is not None:
                if len(map_return_dict[current_map]["annulus_data"]) == 0:
                    print("there is no points in stack to do UNDO plane cutter operation")
                    map_return_dict[current_map]["annulus_status"] = False
                    map_return_dict[current_map]["lr_df_changed"] = False
                else:
                    del map_return_dict[current_map]["annulus_data"][-1]
                    map_return_dict[current_map]["annulus_status"] = True
                    map_return_dict[current_map]["lr_df_changed"] = True


            if len(geo_config) > 0:
                print("geometry parameters are {}".format(geo_config))
                map_return_dict[current_map]["geo_config"] = geo_config
                map_return_dict[current_map]["lr_df_changed"] = True

            if len(ablation_req) > 0:
                acquire_logger.info("ablation point capture starts")
                acquire_logger.info("ablation point req : {}".format(ablation_req))
                acquire_logger.info("ablation point capture ends")

            if len(update_metrics) > 0:
                acquire_logger.info("adjust_activation starts")

                map_return_dict[current_map]["storage"].cummulative_lr_df = \
                    protobuf_packaging.update_metrics_request(update_metrics, map_return_dict[current_map]["storage"])

                map_return_dict[current_map]["lr_df_changed"] = True

                acquire_logger.info("adjust_activation ends")

            if len(update_tag) > 0:
                map_return_dict[current_map]["storage"] = \
                    protobuf_packaging.update_tag(update_tag, map_return_dict[current_map]["storage"])

            if len(plane_cutter) > 0:
                map_return_dict[current_map]["annulus_status"] = True
                map_return_dict[current_map]["annulus_data"].append(plane_cutter)
                map_return_dict[current_map]["lr_df_changed"] = True

            if auto_map_dict["auto_acquire"] or auto_map_dict["auto_geo"] or auto_map_dict["manual_acquire"]:
                map_return_dict[current_map]["annulus_status"] = False
                map_return_dict[current_map]["annulus_data"] = []

            if len(final_geo) > 0:
                print("final geo requst received")
                map_return_dict[current_map]["lr_df_changed"] = True
                map_return_dict[current_map]["final_geo"] = True
            else:
                map_return_dict[current_map]["final_geo"] = False

            if not (auto_map_dict["auto_acquire"]) and auto_map_dict["manual_acquire"]:

                acquire_logger.info("acquire timestamp received : {}".format(auto_map_dict["timestamp"]))

                acquire_received_time = time.time() * 1000

                look_up_df = pd.concat([map_return_dict[current_map]["storage"].manual_beat_df,
                                        map_return_dict[current_map]["storage"].cumulative_beat_df]).reset_index(
                    drop=True)

                if look_up_df.empty:
                    acquire_logger.info("matching beat was not selected")
                    protobuf_packaging.rhythm_change_detected(conn_sender, message="No nearest matching beat",
                                                              code=1005)
                    auto_map_dict["manual_acquire"] = False
                    continue

                index = utils.find_closest_index(look_up_df.timestamp, auto_map_dict["timestamp"])
                auto_map_dict["manual_acquire"] = False

                if abs(look_up_df.loc[index, "end_time"] - auto_map_dict["timestamp"]) <= config[
                    "Threshold_beat_selection"]:
                    look_up_df_matched_df = look_up_df.loc[[index], :]
                    look_up_df_matched_df = look_up_df_matched_df[look_up_df_matched_df["node_name"].isin(
                        config["selected_nodes_for_electrical"])].reset_index(drop=True)

                    look_up_df_matched_df = look_up_df_matched_df[look_up_df_matched_df["electric_beat"] == True]

                    if len(look_up_df_matched_df) > 0:
                        if look_up_df_matched_df.tag_index[0] == -1:
                            if len(map_return_dict[current_map]["storage"].cumulative_beat_df) > 0:
                                start_index = int(
                                    max(map_return_dict[current_map]["storage"].cumulative_beat_df.tag_index))
                            else:
                                start_index = 0

                            look_up_df_matched_df["tag_index"] = list(
                                range(start_index + 1, start_index + 1 + len(look_up_df_matched_df)))

                            # Appending the data to the cumulative beat df
                            map_return_dict[current_map]["storage"].cumulative_beat_df = pd.concat(
                                [map_return_dict[current_map]["storage"].cumulative_beat_df,
                                 look_up_df_matched_df]).reset_index(drop=True)

                            # Appending the data to the LR beat df
                            look_up_df_matched_df["cumulative_beat_tag_index"] = \
                                look_up_df_matched_df["tag_index"]
                        else:
                            look_up_df_matched_df["cumulative_beat_tag_index"] = \
                                look_up_df_matched_df["tag_index"]

                        last_tag_index = len(map_return_dict[current_map]["storage"].cummulative_lr_df)
                        look_up_df_matched_df["tag_index"] = [last_tag_index + X for X in
                                                              range(len(look_up_df_matched_df))]
                        look_up_df_matched_df["auto_acquire"] = False

                        map_return_dict[current_map]["storage"].cummulative_lr_df = pd.concat(
                            [map_return_dict[current_map]["storage"].cummulative_lr_df,
                             look_up_df_matched_df]).reset_index(drop=True)

                        map_return_dict[current_map]["lr_df_changed"] = True
                else:
                    acquire_logger.info("matching beat was not selected")
                    protobuf_packaging.rhythm_change_detected(conn_sender, message="No nearest matching beat",
                                                              code=1005)

            if map_return_dict[current_map]["lr_df_changed"] or auto_map_dict["auto_acquire"] or auto_map_dict[
                "auto_geo"]:

                acquire_logger.info(
                    "lr_df_changed and ndi_df_changed are {} {}".format(map_return_dict[current_map]["lr_df_changed"],
                                                                        map_return_dict[current_map]["ndi_df_changed"]))

                # Geometry construction object is initialised
                acquire_logger.info("Geometry processing starts..")
                geometry_obj = Geometry.Geometry(map_return_dict[current_map]["storage"],
                                                 catheter_obj)
                map_return_dict[current_map] = geometry_obj.complete_process(map_return_dict[current_map],
                                                                             auto_map_dict)
                acquire_logger.info("Geometry processing ends..")

                if len(map_return_dict[current_map]["storage"].computed_metric_df) > 0:
                    acquire_logger.info("send all beats special point protobuf send starts")
                    protobuf_packaging.send_ecg_egm_acquire(
                        map_return_dict[current_map]["storage"].computed_metric_df.copy(),
                        config["MEASURES_REQUIRED"],
                        config["annotation_columns"],
                        conn_sender, map_return_dict[current_map])
                    acquire_logger.info("send all beats special point protobuf send ends")
                # If the number of atleast one new measured point, it is added to the map_return_dict[current_map][
                # "storage"]
                if len(geometry_obj.output_lr_points) > 0:
                    acquire_logger.info("append_coordinate_cummulative_lr_df starts..")
                    map_return_dict[current_map]["storage"].append_coordinate_cummulative_lr_df(
                        geometry_obj.output_lr_points)
                    acquire_logger.info("append_coordinate_cummulative_lr_df ends..")

                # If there is atleast unmeasured points then the interpolation object is created and interpolation is
                # called

                if geometry_obj.output_hr_points.shape[0] > 0:

                    if map_return_dict[current_map]["storage"].cummulative_lr_df.shape[0] > 0:
                        # acquire_logger.info("Interpolation starting....", forced="print")

                        acquire_logger.info("Interpolation starts..")
                        if config["interpolation_type"] == "inverse":
                            interpolation_obj = Interpolation.InverseInterpolation(proc_pool)
                            interpolated_geo_df = interpolation_obj.interpolation(geometry_obj.output_hr_points.copy())
                        elif config["interpolation_type"] == "bob":
                            interpolation_obj = Interpolation.interpolation_bob()
                            interpolated_geo_df = interpolation_obj.bob_interpolation(
                                geometry_obj.output_hr_points.copy(), proc_pool,
                                map_return_dict[current_map]["lr_df_changed"])
                        acquire_logger.info("Interpolation ends..")
                    else:
                        interpolated_geo_df = geometry_obj.output_hr_points.copy()

                    # Sending the interpolated data to the UI layer
                    conn_sender = data_subscriber.reconnect(conn_sender, config["queue_host"], config["queue_port"])

                    isochone_computation = True
                    if auto_map_dict["manual_acquire"]:
                        isochone_computation = False

                    acquire_logger.info("model protobuf conversion - starts")
                    # protobuf_packaging.connect_and_send(interpolated_geo_df.copy(), conn_sender,
                    #                                     map_return_dict[current_map], isochone_computation)
                    protobuf_packaging.dataframe_to_protobuf(interpolated_geo_df.copy(), conn_sender,
                                                        map_return_dict[current_map], isochone_computation)
                    acquire_logger.info("model protobuf conversion - ends")
                    print("model protobuf sent {}".format(current_map))
                    map_return_dict[current_map]["storage"].interpolated_geo_df = interpolated_geo_df.copy()
                    map_return_dict[current_map]["lr_df_changed"] = False
                    auto_map_dict["manual_acquire"] = False

            map_return_dict[current_map]["lr_df_changed"] = False
            map_return_dict[current_map]["ndi_df_changed"] = False
            map_return_dict[current_map]["annulus_status"] = False

        except InterruptedError:
            print("************************InterruptedError**********************")
            crash_back_up()
            pass
        except Exception as e:
            print("Exception encountered")
            # print(str(e))
            # acquire_logger.info(str(e))
            acquire_logger.error(''.join(traceback.format_exc()))
            continue


if __name__ == "__main__":
    main()
