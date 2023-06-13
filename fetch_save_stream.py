"""
 *  @file  fetch_save_stream.py
 *  @brief Fetches the stream from the CP-PLUS CCTVs and save it onto a folder.
 *
 *  @author Kalp Garg.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
import pytz
import cv2
from py_logging import get_logger

global logger


def return_datetime(mode=1):
    date_format = '%Y-%m-%d__%H_%M_%S'
    dnt_utc = datetime.now(tz=pytz.utc)
    dnt_pdt = dnt_utc.astimezone()
    if mode == 0:
        return dnt_pdt
    elif mode == 1:
        return dnt_pdt.strftime(date_format)


def parse_cam_info_json_file(file_location):
    if os.path.exists(file_location):
        with open(file_location) as f:
            cred_json = [line.rstrip('\n') for line in f]
        cred_json = str(cred_json)
        cred_json = "".join(cred_json.split())
        cred_json = cred_json.replace("','", "")
        cred_json = cred_json.replace("['", "")
        cred_json = cred_json.replace("']", "")
        cred_json = json.loads(cred_json)

        u_name = cred_json["CP_PLUS_DVR"]["rtsp_cred"]["UserName"]
        pass_w = cred_json["CP_PLUS_DVR"]["rtsp_cred"]["Password"]
        IP = cred_json["CP_PLUS_DVR"]["rtsp_cred"]["IP"]
        port_no = cred_json["CP_PLUS_DVR"]["rtsp_cred"]["port_number"]
        cam_wid = cred_json["CP_PLUS_DVR"]["resolution"]["Width"]
        cam_hei = cred_json["CP_PLUS_DVR"]["resolution"]["Height"]

        return [u_name, pass_w, IP, port_no, cam_wid, cam_hei]


class FetchStream(object):

    def __init__(self, u_name, u_passwd, IP, port_no, cam_width, cam_height):
        self.rtsp = "rtsp://{}:{}@{}:{}/cam/realmonitor?".format(u_name, u_passwd, IP, port_no)
        self.width = int(cam_width)
        self.height = int(cam_height)
        self.fps = int(30)  #default value

    def connect_camera(self, ch, subtype=0):
        ch_substr = "channel={}&subtype={}".format(ch, subtype)
        new_rtsp = self.rtsp + ch_substr
        cam = cv2.VideoCapture(new_rtsp, cv2.CAP_FFMPEG)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)  # ID number for width is 3
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)  # ID number for height is 4
        # cam.set(cv2.CAP_PROP_FPS, 30)
        # cam.set(10, 10000)  # ID number for brightness is 10
        self.fps = int(cam.get(cv2.CAP_PROP_FPS))
        self.width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        return cam

    def display_save_live_stream(self, cams, log_folder, period, cam_no, save_stream=False):
        pTime = 0
        start_time = time.time()
        start_dt = return_datetime()
        video_codec = cv2.VideoWriter_fourcc('m','p','4','v')
        # video_codec = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')

        if save_stream:
            path_exists = os.path.exists(os.path.join(log_folder, "recordings", "cam{}".format(cam_no)))
            if not path_exists:
                os.makedirs(os.path.join(log_folder, "recordings", "cam{}".format(cam_no)))
            recordings_dir = os.path.join(log_folder, "recordings", "cam{}".format(cam_no))
            logger.info("Recodings dir : {}".format(recordings_dir))
            # Create a video write before entering the loop
            first_v_file = os.path.join(recordings_dir, "{}".format(start_dt)+ ".mp4")
            video_writer = cv2.VideoWriter(
                first_v_file, video_codec, self.fps, (self.width, self.height))

        while True:
            success, current_screen = cams.read()
            frame = current_screen
            # Full_frame = cv2.resize(self.main_screen, dim, interpolation=cv2.INTER_AREA)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, 'FPS: {}'.format(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
            # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("CP_PLUS", frame)
            if save_stream:
                if time.time() - start_time > period:
                    logger.info("Saving stream")
                    end_dt = return_datetime()
                    video_file = os.path.join(recordings_dir, "{}_to_{}".format(start_dt, end_dt) + ".mp4")
                    logger.info('FPS: {}'.format(int(fps)))
                    video_writer = cv2.VideoWriter(video_file, video_codec, self.fps, (self.width, self.height))
                    start_time = time.time()
                    start_dt = end_dt
                if success:
                    video_writer.write(frame)
                else:
                    logger.info("Unable to read from stream.")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cams.release()
                video_writer.release()
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    cam_stream_args = argparse.ArgumentParser(description="Create and save Camera stream")
    cam_stream_args.version = "23.03.01"  # yy.mm.vv
    cam_stream_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    cam_stream_args.add_argument('-l', '--log_folder', type=str, metavar='cam_stream_log',
                                 default="cam_stream_log",
                                 help="Location of the log folder")
    cam_stream_args.add_argument('-cn', '--camera_no', action='store', type= int, default=1, choices = range(1,5),metavar='[1-4]', help='Camera number to stream. Default is 1. Range is 1 to 4')
    cam_stream_args.add_argument('-p', '--time_period', action='store', type=int, default=10,
                                 metavar='10', help='Timeperiod of saving livestream. Default is 10')
    cam_stream_args.add_argument('-cl', '--cred_loc', action='store', metavar='cam_info.json', type=str,
                                 help='path of cam info file', required=True)

    args = cam_stream_args.parse_args()

    addl_file_loc = os.path.join("cam_stream", args.log_folder,
                                 "{}_{}.txt".format("cam_stream_logs_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    logger.info("Script version is: {}".format(cam_stream_args.version))
    [u_name, pass_w, IP, port_no, cam_wid, cam_hei] = parse_cam_info_json_file(args.cred_loc)

    cam_object = FetchStream(u_name, pass_w, IP, port_no, cam_wid, cam_hei)
    cam_no = args.camera_no
    cams = cam_object.connect_camera(cam_no)
    cam_object.display_save_live_stream(cams, args.log_folder, args.time_period, args.camera_no, save_stream=True)
