# This function loads the config file and processes it.
# It returns two things:
#   1., general_settings is a dict, where camera and miscellaneous settings were defined
#   2., markers is an array of dicts, which contains specific settings for the markers themselves.
import configparser as cp #This is needed for reading the config file
import numpy as np #NumPy, because we can
import copy #dicts have mutable variables, I need to clone them.

def process_config_file(config_file_name):
    configs = cp.ConfigParser()
    configs.read(config_file_name) #Read the config file as a dictionary
    config_section_names = configs.sections() #Read in the section names.
    assert config_section_names != 'General', "The 'General' section does not exist in the config file!"
    general_settings = dict()
        #Number of markers
    general_settings["no_of_markers"] = np.int_(len(configs) - 2) #The first entry is the general settings
        #Camera ID
    general_settings["camera_id"] = np.int_(configs['General']['camera_id']) #We need to know which camera to use.
        #Camera focal lengths for camera matrix
    general_settings["camera_focal_length_x"] = np.float_(configs['General']['camera_focal_length_x'])
    general_settings["camera_center_x"] = np.float_(configs['General']['camera_center_x'])
    general_settings["camera_focal_length_y"] = np.float_(configs['General']['camera_focal_length_y'])
    general_settings["camera_center_y"] = np.float_(configs['General']['camera_center_y'])
        #Camera distortion coefficients
    general_settings["camera_k1"] = np.float_(configs['General']['camera_dist_coeff_k1'])
    general_settings["camera_k2"] = np.float_(configs['General']['camera_dist_coeff_k2'])
    general_settings["camera_p1"] = np.float_(configs['General']['camera_dist_coeff_p1'])
    general_settings["camera_p2"] = np.float_(configs['General']['camera_dist_coeff_p2'])
    general_settings["camera_k3"] = np.float_(configs['General']['camera_dist_coeff_k3'])
        #Display camera image
    general_settings["camera_show_picture"] = np.int_(configs['General']['camera_show_picture'])
        #Server port, converted to an integer, not a script.
    general_settings["server_udp_port"] = np.int_(configs['General']['server_udp_port'])
        
#print("Config file read and processed.")
    #Return the variables
    return general_settings
