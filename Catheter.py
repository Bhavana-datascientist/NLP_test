import itertools
from itertools import repeat
from config_dict import config
import numpy as np
import pandas as pd
from PyEMD import EMD
emd = EMD()
import logging
acquire_logger = logging.getLogger("acquisition")

class Catheter:
    def __init__(self):
        self.no_nodes = config["N_nodes"]
        self.distance_between_nodes = config["distance_between_nodes"]
        self.NDI_coordinates = config["NDI_coordinates"]
        self.Quaternions = config["Quaternions"]
        self.respiration_columns = ["timestamp"] + self.NDI_coordinates +["raw_x","raw_y","raw_z"]+ self.Quaternions

    @staticmethod
    def signalavg1(self, idx, dataset):
        signal_avg1 = sum(dataset[:idx + 1]) / (idx + 1)
        return signal_avg1

    @staticmethod
    def signalavg2(idx, dataset):
        signal_avg2 = sum(dataset[(idx + 1) - 40:idx + 1]) / (idx + 1)
        return signal_avg2

    def respiration_correction(self, df):
        acquire_logger.info("respiration_correction starts")
        for i in range(df.shape[0]):
            if i < 40:
                df.at[i, self.NDI_coordinates] = np.mean(df.loc[:i, self.NDI_coordinates])
            elif i >= 40:
                df.at[i, self.NDI_coordinates] = np.mean(df.loc[(i-40, i), self.NDI_coordinates])
        acquire_logger.info("respiration_correction ends")
        return df

    def respiration_correction_method(self,df,return_dict):
        acquire_logger.info("respiration_correction starts")
        return_dict["last_n_samples"] = len(return_dict["acumulated_coordinates"])

        if return_dict["acumulated_coordinates"].size > 0:
            return_dict["acumulated_coordinates"] = np.append(return_dict["acumulated_coordinates"],
                                                              df[['Tx', 'Ty', 'Tz']].to_numpy(), axis=0)
        else:
            return_dict["acumulated_coordinates"] = df[['Tx', 'Ty', 'Tz']].to_numpy()

        return_dict["n_samples"] = len(return_dict["acumulated_coordinates"])

        if return_dict["n_samples"] <= 50:
            for i in range(return_dict["last_n_samples"], return_dict["n_samples"]):
                return_dict["acumulated_coordinates"][i, :] = np.mean(return_dict["acumulated_coordinates"][:i + 1, :],
                                                                      axis=0)

        elif (return_dict["n_samples"] > 50) & (return_dict["last_n_samples"]>=2):
            acquire_logger.info("respiration_correction for more than 50 points starts")
            return_dict["acumulated_coordinates"][return_dict["last_n_samples"]:return_dict["n_samples"], 0] = emd(return_dict["acumulated_coordinates"][:, 0])[-1][return_dict["last_n_samples"]:return_dict["n_samples"]]
            return_dict["acumulated_coordinates"][return_dict["last_n_samples"]:return_dict["n_samples"], 1] = emd(return_dict["acumulated_coordinates"][:, 1])[-1][return_dict["last_n_samples"]:return_dict["n_samples"]]
            return_dict["acumulated_coordinates"][return_dict["last_n_samples"]:return_dict["n_samples"], 2] = emd(return_dict["acumulated_coordinates"][:, 2])[-1][return_dict["last_n_samples"]:return_dict["n_samples"]]
            acquire_logger.info("respiration_correction for more than 50 points ends")

        df[['x','y','z']] = pd.DataFrame(return_dict["acumulated_coordinates"][-df.shape[0]:, :],columns=['x','y','z'],index=df.index)
        return_dict["acumulated_coordinates"] = return_dict["acumulated_coordinates"][-1000:, :]
        acquire_logger.info("respiration_correction ends")
        return df

    @staticmethod
    def quaternion_to_euler(qw, qx, qy, qz):
        """Converts quaternion to euler angles"""
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx ** 2 + qy ** 2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.where(np.abs(sinp) >= 1,
                         np.sign(sinp) * np.pi / 2,
                         np.arcsin(sinp))

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy ** 2 + qz ** 2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def rotation_matrix_from_quaternions(qw, qx, qy, qz):
        """Get rotation matrix from quaternions """

        first = [qw**2 + qx ** 2 - qy ** 2 - qz ** 2, 2*qx*qy - 2*qw*qz, 2*qx*qz + 2*qw*qy]
        second = [2*qx*qy + 2*qw*qz, qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2, 2*qy*qz - 2*qw*qx]
        third = [2*qx*qz - 2*qw*qy, 2*qy*qz + 2*qw*qx, qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2]
        R = np.array([first, second, third])
        if config['Catheter_type'] == 'TactiCath':
                Rotated_matrix = R.dot((np.asarray([0, 0, -1])))
        else:
            Rotated_matrix = R.dot((np.asarray([0, 0, 1])))
        return Rotated_matrix

    def direction_transform(self, message, distance_between_nodes, node_names):
        """
        :param vect: 3D-point co-ordinates
        :param q_vector: Quaternions
        :param distance_between_nodes: distance_between_nodes from config
        :return: D 4*3D point co-ordinates
        """
        time_stamp = float(message[0])
        vect = np.array([message[1], message[2], message[3]])
        raw_vect = np.array([message[4], message[5], message[6]])
        qw = message[7]
        qx = message[8]
        qy = message[9]
        qz = message[10]
        vect2 = self.rotation_matrix_from_quaternions(qw, qx, qy, qz)

        dist_vector_data = []
        if len(distance_between_nodes) == len(node_names):
            for dist, node in zip(distance_between_nodes, node_names):
                dist_vect = vect - (dist * vect2)
                raw_dist_vect = raw_vect - (dist * vect2)
                dist_vector_data.append([time_stamp, *dist_vect, *raw_dist_vect,node,1])

        return dist_vector_data

    def convert_tip_to_nodes(self, proc_pool, coordinate_messages_df, return_dict):

        coordinate_messages = coordinate_messages_df[self.respiration_columns].values.tolist()

        node_names = config["EGM_columns"]

        tip_to_node_converted_coordinates = proc_pool.starmap(self.direction_transform,
                                                              zip(coordinate_messages,
                                                                  repeat(self.distance_between_nodes),
                                                                  repeat(node_names)))

        df = pd.DataFrame(list(itertools.chain(*tip_to_node_converted_coordinates)), columns=config["NDI_df_columns"])

        return df


