import os, json


def parse_json(file):
    with open(file) as f:
        cred_json = [line.rstrip('\n') for line in f]
    cred_json = str(cred_json)
    cred_json = "".join(cred_json.split())
    cred_json = cred_json.replace("','", "")
    cred_json = cred_json.replace("['", "")
    cred_json = cred_json.replace("']", "")
    cred_json = json.loads(cred_json)
    return cred_json


def get_cam_info(file_location):
    if os.path.exists(file_location):
        cred_json = parse_json(file_location)
        u_name = cred_json["CP_PLUS_DVR"]["rtsp_cred"]["UserName"]
        pass_w = cred_json["CP_PLUS_DVR"]["rtsp_cred"]["Password"]
        IP = cred_json["CP_PLUS_DVR"]["rtsp_cred"]["IP"]
        port_no = cred_json["CP_PLUS_DVR"]["rtsp_cred"]["port_number"]
        cam_wid = cred_json["CP_PLUS_DVR"]["resolution"]["Width"]
        cam_hei = cred_json["CP_PLUS_DVR"]["resolution"]["Height"]

        return [u_name, pass_w, IP, port_no, cam_wid, cam_hei]


def get_cropped_params(file_location, cam_no):
    if os.path.exists(file_location):
        cred_json = parse_json(file_location)
        cam_type = "cam{}".format(cam_no)
        start_x = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["start_x"]
        start_y = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["start_y"]
        width_x = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["width_x"]
        width_y = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["height_y"]

        return [start_x, start_y, width_x, width_y]
