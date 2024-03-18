import os, json
from datetime import datetime, timedelta
import pytz, subprocess

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

def return_basepath():
    '''
    Will return git top level path if exists otherwise location of this script
    '''
    try:
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        git_root_path = result.stdout.strip()
        # print("Git repository root path:", git_root_path)
        return git_root_path
    except subprocess.CalledProcessError:
        print("Not a Git repository or an error occurred. Returning path of this script")
        script_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Path of this script is: {script_path}")
        return script_path

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


def get_cropped_params(file_location, cam_no, extract_type="rectangle"):
    if os.path.exists(file_location):
        cred_json = parse_json(file_location)
        cam_type = "cam{}".format(cam_no)
        if extract_type == "rectangle":
            start_x = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["rectangle"]["start_x"]
            start_y = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["rectangle"]["start_y"]
            width_x = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["rectangle"]["width_x"]
            width_y = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["rectangle"]["height_y"]
            return [start_x, start_y, width_x, width_y]
        elif extract_type == "polygon":
            vertices=[]
            size = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["polygon"]["size"]
            points_x = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["polygon"]["points_x"]
            points_y = cred_json["CP_PLUS_DVR"]["crop_details"][cam_type]["polygon"]["points_y"]
            for i in range(size):
                vertices.append((points_x[i], points_y[i]))

            return vertices
        else:
            print("Extraction type doesn't exist. Choose either from polygon or rectangle")
            quit()

def get_cam_loc(file_location, cam_no):
    if os.path.exists(file_location):
        cred_json = parse_json(file_location)
        cam_type = "cam{}".format(cam_no)
        return cred_json["CP_PLUS_DVR"]["cam_loc_details"][cam_type]


def get_secret_key(file_location):
    if os.path.exists(file_location):
        cred_json = parse_json(file_location)
        secret_key = cred_json["KEYS"]["SECRET_KEY"]
        return secret_key

def get_tgram_keys(file_location):
    if os.path.exists(file_location):
        cred_json = parse_json(file_location)
        tgram_bot_id = cred_json["KEYS"]["HOME_CAM_BOT_ID"]
        home_recor_chat_id = cred_json["KEYS"]["HOME_RECORDINGS_CHAT_ID"]
        return (tgram_bot_id, home_recor_chat_id)

def return_datetime(mode=1, period=None):
    date_format = '%Y-%m-%d__%H_%M_%S'
    dnt_utc = datetime.now(tz=pytz.utc)
    dnt_pdt = dnt_utc.astimezone()
    if mode == 0:
        return dnt_pdt
    elif mode == 1:
        return dnt_pdt.strftime(date_format)
    elif mode == 2:
        delta_time = dnt_pdt + timedelta(seconds=period)
        return delta_time.strftime(date_format)

def return_start_end_dnt(f_name):
    date_format = '%Y-%m-%d__%H_%M_%S'
    try:
        [start_date, end_date] = f_name.split("_to_")
        return datetime.strptime(start_date, date_format), datetime.strptime(end_date, date_format)
    except Exception as e:
        start_date = f_name
        start_dnt = datetime.strptime(start_date, date_format)
        end_dnt = start_dnt + timedelta(seconds=15)
        return start_dnt, end_dnt
    
def write_to_file(fpath, write_msg):
    with open(fpath, "w") as f1:
        f1.write(write_msg)
    
def read_file(fpath):
    with open(fpath, "r") as f1:
        return f1.read()
