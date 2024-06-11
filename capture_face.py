"""
 *  @file  capture_face.py
 *  @brief Captures the face from webcam input and put them into a corresponding folder.
 *
 *  @author Kalp Garg.
"""
import os
import time

import cv2
import argparse

from py_logging import get_logger
from datetime import datetime
# from cvzone.SelfiSegmentationModule import SelfiSegmentation
from common_utils import get_cropped_params
import pytz
import uuid


def return_datetime(mode=1):
    date_format = '%Y-%m-%d__%H_%M_%S'
    dnt_utc = datetime.now(tz=pytz.utc)
    dnt_pdt = dnt_utc.astimezone()
    if mode == 0:
        return dnt_pdt
    elif mode == 1:
        return dnt_pdt.strftime(date_format)


class CaptureFace(object):
    def __init__(self, source):
        self.cam_source = source

    def connect_camera(self):
        cam = cv2.VideoCapture(self.cam_source)
        return cam

    def create_dir(self, log_folder, name):
        if not os.path.exists(os.path.join(log_folder, "input_db", str(name))):
            os.makedirs(os.path.join(log_folder, "input_db", str(name)))

        return os.path.join(log_folder, "input_db", str(name))

    def do_resizing(self, input_img):

        # Resize the image while preserving the aspect ratio
        target_size = (224, 224)
        resized_image = cv2.resize(input_img, target_size, interpolation=cv2.INTER_AREA)

        return resized_image

    def save_capture(self, log_folder, name, total_capture_cnt, sleep_t, cred_loc, cam_no):
        capture_from_stream = False
        sleep_time = (sleep_t) / 1000
        cam = self.connect_camera()
        total_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cam.get(cv2.CAP_PROP_FPS)
        logger.info("Total frames : {}. FPS : {}".format(total_frames, fps))
        if total_frames == 0:
            capture_from_stream = True
        if not capture_from_stream:
            if fps == 0:
                logger.error("FPS can't be zero. PLS CHECK...")
                return 0
            total_time = (total_frames / fps)
            if total_capture_cnt == 0:
                logger.error("Total capture cnt can't be zero. PLS CHECK... ")
                return 0
            sleep_time = round((total_time / total_capture_cnt), 2)
        dir_path = self.create_dir(log_folder, "raw_untouched_data")
        logger.info("Directory path : {}".format(dir_path))
        # segmentor = SelfiSegmentation()
        logger.info("Time between capture is : {} s".format(sleep_time))
        for i in range(total_capture_cnt):
            if cam.isOpened():
                if not capture_from_stream:
                    t_msec = i * sleep_time * 1000
                    cam.set(cv2.CAP_PROP_POS_MSEC, t_msec)
                else:
                    time.sleep(sleep_time)
                success, img = cam.read()
                # start_time = time.time()
                if success:
                    # bg_rem_img = segmentor.removeBG(img, (255, 255, 255), threshold=0.1)
                    cam_used = 1
                    [start_x, start_y, width_x, height_y] = get_cropped_params(cred_loc, cam_no)
                    cropped_image = img[start_y:start_y + height_y, start_x:start_x + width_x]
                    resized_image = self.do_resizing(cropped_image)
                    # cv2.imshow("original", resized_image)
                    rand_uuid = uuid.uuid1()
                    img_path = os.path.join(dir_path, "{}{}_{}".format(name, i, rand_uuid) + ".jpg")
                    cv2.imwrite(img_path, resized_image)
                    logger.info("File {} successfully written".format(img_path))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break
        cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    capture_face_args = argparse.ArgumentParser(description="Create and save Camera stream")
    capture_face_args.version = "23.03.01"  # yy.mm.vv
    capture_face_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    capture_face_args.add_argument('-s', '--source', action='store', type=str, default='webcam', metavar='webcam',
                                   help='Source of capture. Default is webcam')
    capture_face_args.add_argument('-l', '--log_folder', type=str, metavar='capture_face',
                                   default="capture_face_log",
                                   help="Location of the log folder")
    capture_face_args.add_argument('-n', '--no_of_capture', action='store', type=int, default=10,
                                   metavar='10', help='Number of captures. Default is 10.')
    capture_face_args.add_argument('-t', '--time_bw_capture', action='store', type=int, default=1,
                                   metavar='1',
                                   help='Time(in seconds) between two captures. Not used for pre-recorded videos')
    capture_face_args.add_argument('-p', '--person_name', action='store', type=str, required=True,
                                   metavar='person1', help='person name')
    capture_face_args.add_argument('-d', '--is_dir', action='store', type=bool, default=False, metavar='False',
                                   help='Whether source is a directory which contain multiple videos')
    capture_face_args.add_argument('-cn', '--camera_no', action='store', type=int, choices=range(1, 5),
                                 metavar='[1-4]', help='Camera number to stream. Default is 1. Range is 1 to 4', required=True)
    capture_face_args.add_argument('-cl', '--cred_loc', action='store', metavar='cam_info.json', type=str,
                                 help='path of cam info file', required=True)

    args = capture_face_args.parse_args()

    addl_file_loc = os.path.join("capture_face", args.log_folder,
                                 "{}_{}.txt".format("capture_face_logs_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=False)
    logger.info("Script version is: {}".format(capture_face_args.version))

    if args.source == "webcam":
        source = 0
    else:
        source = args.source

    if args.is_dir:
        if not os.path.exists(source):
            logger.error("Given path {} doesn't exist. Please check.. ".format(source))
            exit()
        for video in os.listdir(source):
            cam_object = CaptureFace(os.path.join(source, video))
            cam_object.save_capture(args.log_folder, args.person_name, args.no_of_capture, args.time_bw_capture,
                                    args.cred_loc, args.camera_no)
    else:
        cam_object = CaptureFace(source)
        cam_object.save_capture(args.log_folder, args.person_name, args.no_of_capture, args.time_bw_capture,
                                args.cred_loc, args.camera_no)
