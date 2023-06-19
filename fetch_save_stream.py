"""
 *  @file  fetch_save_stream.py
 *  @brief Fetches the stream from the CP-PLUS CCTVs and save it onto a folder.
 *
 *  @author Kalp Garg.
"""
import argparse
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pytz
import cv2
from py_logging import get_logger
from common_utils import get_cam_info
from common_utils import get_cropped_params
import threading

global logger


def return_datetime(mode=1, period=None):
    date_format = '%Y-%m-%d__%H_%M_%S'
    dnt_utc = datetime.now(tz=pytz.utc)
    dnt_pdt = dnt_utc.astimezone()
    if mode == 0:
        return dnt_pdt
    elif mode == 1:
        return dnt_pdt.strftime(date_format)
    elif mode ==2:
        delta_time = dnt_pdt + timedelta(seconds=period)
        return delta_time.strftime(date_format)

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

    def display_save_live_stream(self, cams, log_folder, period, cam_no, motion_detection, cred_loc, save_stream=False):
        pTime = 0
        retry_limit = 10
        retry_count = 0
        cntr_save_stream = time.time()
        motion_detect_time = time.time()
        upd_start_frame = True
        motion_detected = True
        prev_capture_running = False
        motion_alarm_cntr = 0
        start_dt = return_datetime()
        video_codec = cv2.VideoWriter_fourcc('m','p','4','v')
        cropped_vertices = get_cropped_params(cred_loc, cam_no, extract_type="polygon")
        # video_codec = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')

        if save_stream:
            path_exists = os.path.exists(os.path.join(log_folder, "recordings", "cam{}".format(cam_no)))
            if not path_exists:
                os.makedirs(os.path.join(log_folder, "recordings", "cam{}".format(cam_no)))
            recordings_dir = os.path.join(log_folder, "recordings", "cam{}".format(cam_no))
            logger.info("Recodings dir : {}".format(recordings_dir))

            # Create a video writer instance before entering the loop
            old_video_file = os.path.join(recordings_dir, "{}".format(start_dt)+ ".mp4")
            video_writer = cv2.VideoWriter(
                old_video_file, video_codec, self.fps, (self.width, self.height))

        while True:
            success, current_screen = cams.read()
            frame = current_screen

            # Full_frame = cv2.resize(self.main_screen, dim, interpolation=cv2.INTER_AREA)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, 'FPS: {}'.format(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0),
                        2)
            # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("CP_PLUS", frame)
            if motion_detection:
                if not prev_capture_running:
                    # Do this only first time.
                    if upd_start_frame:
                        start_frame = frame
                        # Do cropping, convert it to grayscale and apply gaussian kernel
                        # Create an empty mask with the same dimensions as the image
                        mask = np.zeros_like(start_frame)

                        # Fill the polygon region in the mask with white (255) pixels
                        cv2.fillPoly(mask, [np.array(cropped_vertices)], (255, 255, 255))

                        # Apply the mask to the image to extract the region of interest
                        start_frame = cv2.bitwise_and(start_frame, mask)
                        start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
                        start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)
                        upd_start_frame = False

                    # Convert curr_frame to grayscale
                    # Create an empty mask with the same dimensions as the image
                    mask = np.zeros_like(frame)

                    # Fill the polygon region in the mask with white (255) pixels
                    cv2.fillPoly(mask, [np.array(cropped_vertices)], (255, 255, 255))

                    # Apply the mask to the image to extract the region of interest
                    curr_bw_frame = cv2.bitwise_and(frame, mask)
                    curr_bw_frame = cv2.cvtColor(curr_bw_frame, cv2.COLOR_BGR2GRAY)
                    curr_bw_frame = cv2.GaussianBlur(curr_bw_frame, (5, 5), 0)

                    # Calculate the difference b/w two frames
                    difference = cv2.absdiff(curr_bw_frame, start_frame)
                    threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
                    start_frame = curr_bw_frame

                    if threshold.sum() > 50000:
                        # 2,00,00,000 is the max value
                        print(threshold.sum())
                        motion_alarm_cntr += 1
                    else:
                        if motion_alarm_cntr > 0:
                            motion_alarm_cntr -= 1
                    # cv2.imshow("diff", start_frame)

                    if motion_alarm_cntr > 20:
                        print(motion_alarm_cntr)
                        motion_detected = True
                        logger.info("Motion detected")
                        motion_detect_time = time.time()
                        start_dt = return_datetime()
                        end_dt = return_datetime(mode=2, period=period)
                        new_video_file = os.path.join(recordings_dir, "{}_to_{}".format(start_dt, end_dt) + ".mp4")
                        logger.info("Saving stream to loc: {}".format(new_video_file))
                        # retry this till retry limit
                        while retry_count < retry_limit:
                            try:
                                # Attempt to create the video writer
                                video_writer = cv2.VideoWriter(new_video_file, video_codec, self.fps,
                                                               (self.width, self.height), isColor=True)
                                break
                            except cv2.error as e:
                                print(str(e))
                                if 'codec mpeg4' in str(e):
                                    # Retry if timebase error occurs
                                    retry_count += 1
                                    print(f"Timebase error occurred. Retrying... Attempt {retry_count}/{retry_limit}")
                                else:
                                    # Other errors, handle or re-raise as needed
                                    raise e
                        motion_alarm_cntr = 0
                    else:
                        motion_detected = False

                if success:
                    if motion_detected:
                        if time.time() - motion_detect_time < period:
                            prev_capture_running = True
                        else:
                            prev_capture_running = False
                        if save_stream:
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
    cam_stream_args.add_argument('-md', '--motion_detection', action='store', metavar='True', type=bool,
                                 help='saves video having motion', default=True)

    args = cam_stream_args.parse_args()

    addl_file_loc = os.path.join("cam_stream", args.log_folder,
                                 "{}_{}.txt".format("cam_stream_logs_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    logger.info("Script version is: {}".format(cam_stream_args.version))
    [u_name, pass_w, IP, port_no, cam_wid, cam_hei] = get_cam_info(args.cred_loc)

    cam_object = FetchStream(u_name, pass_w, IP, port_no, cam_wid, cam_hei)
    cam_no = args.camera_no
    cams = cam_object.connect_camera(cam_no)
    cam_object.display_save_live_stream(cams, args.log_folder, args.time_period, args.camera_no, args.motion_detection, args.cred_loc, save_stream=True)
