"""
 *  @file  face_detect.py
 *  @brief Parses already saved videos and detect faces. Delete them after parsing is done.
 *
 *  @author Kalp Garg.
"""
import argparse
import json
import os
import shutil
import random
import time
from datetime import datetime, timezone
import pytz
import cv2
import tensorflow as tf
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
from deepface import DeepFace
from py_logging import get_logger
import albumentations as alb

global logger


def return_datetime(mode=1):
    date_format = '%Y-%m-%d__%H_%M_%S'
    dnt_utc = datetime.now(tz=pytz.utc)
    dnt_pdt = dnt_utc.astimezone()
    if mode == 0:
        return dnt_pdt
    elif mode == 1:
        return dnt_pdt.strftime(date_format)


class FaceDetect(object):
    def __init__(self, in_folder):
        self.in_dir = in_folder
        self.input_db_path = "/Users/kgarg/Documents/extras/home_cam_security/input/input_db/"
        self.parse_in_dir()

    def parse_in_dir(self):
        if not os.path.exists(self.in_dir):
            logger.error("Given path {} doesn't exist. Please check.. ".format(self.in_dir))
            return 0
        total_files =len(os.listdir(self.in_dir))
        return os.listdir(self.in_dir)

    def video_processing_pipeline(self):
        current_dir = self.parse_in_dir()
        i_time = time.time()
        for id, video in enumerate(current_dir):
            video_path = os.path.join(self.in_dir, video)
            logger.info("Processing video : {}".format(video))
            i_time = time.time()
            # self.face_detect_and_identify(video_path)
            self.deepface_stream(video_path)
            logger.info("Time to process : {}".format(time.time() - i_time))

    def deepface_stream(self, v_path):
        input_db_path = "/Users/kgarg/Documents/extras/home_cam_security/input/input_db"
        DeepFace.stream(db_path= input_db_path, enable_face_analysis=False, frame_threshold=1, time_threshold=1, detector_backend="opencv", source=v_path)

    def limit_gpu_growth(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU: {}".format(tf.config.list_physical_devices('GPU')))

    def load_images(self):
        self.limit_gpu_growth()
        data_dir_path = self.input_db_path + "kalp"
        images = tf.data.Dataset.list_files("{}/*.jpg".format(data_dir_path), shuffle=False)

        # print(images.as_numpy_iterator().next())
        images = images.map(self.load_image)

        # print(images.as_numpy_iterator().next())
        # print(type(images))

    def create_train_test_val_split(self):
        data_dir_path = self.input_db_path + "kalp"
        total_files = len(os.listdir(data_dir_path))
        # print(os.listdir(data_dir_path), total_files)
        total_training_sample = int(0.7*total_files)
        total_test_sample = int(0.16*total_files)
        total_val_sample = int(0.14*total_files)

        logger.info("Total \n training sample: {} \n test_sample: {} \n validation_sample: {}".format(total_training_sample, total_test_sample, total_val_sample))
        # self.segregrate_data(data_type="train", total_data_sample=total_training_sample)
        # self.segregrate_data(data_type="test", total_data_sample=total_test_sample)
        # self.segregrate_data(data_type="val", total_data_sample=total_val_sample)

    def segregrate_data(self, data_type=None, total_data_sample=0):
        if (data_type is None) | (total_data_sample == 0):
            logger.error("Total data sample can't be zero. data type is required. Can be either 'test', 'train' or 'validate'.")
            return 0
        #create dir if not exists
        data_type = str(data_type)
        if not os.path.exists(os.path.join(self.input_db_path, data_type, "data")):
            os.makedirs(os.path.join(self.input_db_path, data_type, "data"))
        if not os.path.exists(os.path.join(self.input_db_path, data_type, "labels")):
            os.makedirs(os.path.join(self.input_db_path, data_type, "labels"))
        data_dir_path = self.input_db_path + "kalp"
        for i in range(total_data_sample):
            data_files = os.listdir(data_dir_path)
            total_files = len(data_files)
            rand_int = random.randint(0, total_files-1)
            logger.info("Total data files: {}. random file: {}".format(total_files, rand_int))
            #put this data into training dir
            existing_fpath = os.path.join(data_dir_path, data_files[rand_int])
            new_fpath = os.path.join(self.input_db_path, data_type, "data", data_files[rand_int])
            os.replace(existing_fpath, new_fpath)

            #put correspondning label into label dir
            f_name = data_files[rand_int].split('.')[0] + '.json'
            existing_fpath = os.path.join(self.input_db_path,"kalp_labels", f_name)
            if os.path.exists(existing_fpath):
                new_fpath = os.path.join(self.input_db_path, data_type, "labels", f_name)
                os.replace(existing_fpath, new_fpath)
                # shutil.copyfile(existing_fpath, new_fpath)

    
    def load_image(self, x):
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        return img

    def face_detect_and_identify(self, v_path):
        cap = cv2.VideoCapture(v_path)
        face_detect_backend = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
        pTime = 0
        count = 1
        segmentor = SelfiSegmentation()
        fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        while cap.isOpened():
            success, img = cap.read()
            if success:
                new_size = (int(960/2), int(1080/2))
                img = cv2.resize(img, new_size)
                fgmask = fgbg.apply(img)
                blur = cv2.GaussianBlur(fgmask, (9, 9), 0)
                ret, gaus_fgmask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # gaus_fgmask = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
                # gaus_fgmask = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,
                #                                     1)
                # cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
                # bg_rem_img = segmentor.removeBG(img, (255,0,0), threshold=0.1)
                bg_rem_img = cv2.bitwise_and(img, img, mask=gaus_fgmask)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                # cv2.putText(img, 'FPS: {}'.format(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                detections = DeepFace.extract_faces(img, new_size, face_detect_backend[2], enforce_detection=False)
                confidence = str(round(detections[0]['confidence']*100, 2)) + "%"
                x,y,w,h = detections[0]['facial_area']['x'], detections[0]['facial_area']['y'], detections[0]['facial_area']['w'], detections[0]['facial_area']['h']
                logger.info("Count : {}, FPS : {}, Confidence: {}, x, y, w, h : {}, {}, {}, {}".format(count, fps, confidence, x, y, w, h))
                cv2.putText(img, confidence, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
                # if confidence > 0.90:
                #     print("1111") # draw rectangle to main image
                count = count + 1
                cv2.imshow("CP_PLUS", img)
                # cv2.imshow("FG_MASK", bg_rem_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detect_args = argparse.ArgumentParser(description="Parse videos and detect face")
    face_detect_args.version = "23.03.01"  # yy.mm.vv
    face_detect_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    face_detect_args.add_argument('-l', '--log_folder', type=str, metavar='face_detect_log',
                                 default="face_detect_log",
                                 help="Location of the log folder")
    # face_detect_args.add_argument('-cn', '--camera_no', action='store', type= int, default=1, choices = range(1,5),metavar='[1-4]', help='Camera number to stream. Default is 1. Range is 1 to 4')
    # face_detect_args.add_argument('-p', '--time_period', action='store', type=int, default=10,
    #                              metavar='10', help='Timeperiod of saving livestream. Default is 10')
    # face_detect_args.add_argument('-cl', '--cred_loc', action='store', metavar='cam_info.json', type=str,
                                 # help='path of cam info file', required=True)
    face_detect_args.add_argument('-il', '--input_log_folder', type=str, metavar='input_log_folder', required=True,
                                 help="Location of the input folder which has streams")

    args = face_detect_args.parse_args()

    addl_file_loc = os.path.join("face_detect", args.log_folder,
                                 "{}_{}.txt".format("face_detect_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    logger.info("Script version is: {}".format(face_detect_args.version))

    face_detect = FaceDetect(in_folder=args.input_log_folder)
    # face_detect.video_processing_pipeline()
    face_detect.load_images()
    face_detect.create_train_test_val_split()

