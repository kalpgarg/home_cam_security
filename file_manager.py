"""
 *  @file  file_manager.py
 *  @brief Monitors recordings directories and manages database by adding and deleting entries
 *
 *  @author Kalp Garg.
"""
import sqlite3
import argparse
import os, time
from common_utils import return_datetime, get_cam_loc, return_start_end_dnt
from py_logging import get_logger

global logger

import pathlib

base_path = pathlib.Path(__file__).parent.resolve()

class Publisher(object):
    def __init__(self, db_path):
        self.main_db = sqlite3.connect(db_path)
        logger.info("Connected to DB successfully")

    def get_directories_loc(self, main_recording_dir, cam_list):
        directories = []
        for cam_no in cam_list:
            cam_recording_path = os.path.join(main_recording_dir, "cam{}".format(cam_no))
            if not os.path.exists(cam_recording_path):
                logger.error(
                    "Given recordings directory {} doesn't exist.Please check.. Quitting...".format(cam_recording_path))
                quit()
            directories.append(cam_recording_path)
        logger.info("Directories to look: {}".format(directories))
        return directories

    def start_publishing(self, directory_list, cred_loc):
        processed_files = []
        for i in range(len(directory_list)):
            processed_files.append(set())
        while True:
            for i, directory in enumerate(directory_list):
                # Get the list of files in the directory
                files = os.listdir(directory)
                files = [os.path.join(directory, f) for f in files]
                files.sort(key=lambda x: os.path.getmtime(x))
                cam_no_str = os.path.split(directory)[1]
                cam_no = int(cam_no_str.replace("cam", ""))
                cam_loc = get_cam_loc(cred_loc, cam_no)
                # Keep track of the files already processed
                new_files = set(files) - processed_files[i]
                # Add db entry for each new file
                for file in new_files:
                    file_path = file
                    if file_path.endswith('.mp4'):
                        # modified_time = os.path.getmtime(file_path)
                        # current_time = time.time()
                        # time_diff = current_time - modified_time
                        # logger.info("file_path: {}. Modified_time: {}".format(file_path, modified_time))
                        # logger.info("Time diff is: {}".format(time_diff))

                        message = f"New file added: {file_path}"
                        logger.info(message)
                        f_name = (os.path.split(file_path)[1]).split(".")[0]
                        start_date, end_date = return_start_end_dnt(f_name)
                        logger.info("start_date: {}. end_date: {}".format(start_date, end_date))
                        full_file_path = os.path.join(base_path, file_path)
                        last_row = self.main_db.execute("SELECT * FROM recordings ORDER BY id DESC LIMIT 1;").fetchone()
                        if last_row is None:
                            index = 1
                        else:
                            index = last_row[1] + 1
                        try:
                            self.main_db.execute("INSERT INTO recordings (index_record,cam_no,file_path,cam_loc,from_dnt,to_dnt) \
                                  VALUES ({}, {},'{}','{}','{}','{}');".format(index, cam_no, full_file_path, cam_loc,start_date,end_date))
                            self.main_db.commit()
                        except Exception as e:
                            if "UNIQUE constraint" not in str(e):
                                raise
                            else:
                                logger.info("Entry {} already exist".format(file_path))
                        processed_files[i].add(file)
                        # cntr = 0

                for file in files:
                    file_path = os.path.join(directory, file)
                    if file_path.endswith('.mp4'):
                        modified_time = os.path.getmtime(file_path)
                        current_time = time.time()
                        time_diff = current_time - modified_time
                        if time_diff >= 7 * 24 * 60 * 60:  # if file is older than 7 days, delete it
                            try:
                                # Delete the file and remove its entry
                                self.main_db.execute("DELETE from recordings where file_path = '{}';".format(os.path.join(base_path, file_path)))
                                self.main_db.commit()
                                os.remove(file_path)
                                logger.info(f"File '{file_path}' deleted successfully.")
                            except FileNotFoundError:
                                logger.error(f"File '{file_path}' not found.")
                            except PermissionError:
                                logger.error(f"Permission denied: unable to delete file '{file_path}'.")
                            except Exception as e:
                                logger.error(f"An error occurred while deleting the file: {str(e)}")

    def __del__(self):
        self.main_db.close()


if __name__ == '__main__':
    file_manager_args = argparse.ArgumentParser(description="Look for new video stream addition and publishes it "
                                                )
    file_manager_args.version = "23.03.01"  # yy.mm.vv
    file_manager_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    file_manager_args.add_argument('-l', '--log_folder', type=str, metavar='file_manager_log',
                                   default="file_manager_log",
                                   help="Location of the log folder")
    file_manager_args.add_argument('-cn', '--camera_no', action='store', type=list, default=[1],
                                   metavar='123', help='Camera recording to publish. Default is 1. Range is 1 to 4')
    file_manager_args.add_argument('-db', '--db_path', action='store', metavar='instance/user_db.db', type=str,
                                   help='path of sqlite db', required=True)
    file_manager_args.add_argument('-cl', '--cred_loc', action='store', metavar='cam_info.json', type=str,
                                 help='path of cam info file', required=True)
    file_manager_args.add_argument('-if', '--recordings_dir', action='store', metavar='cam_stream_log/recordings',
                                   type=str,
                                   help='path of recordings directory', required=True)

    args = file_manager_args.parse_args()

    addl_file_loc = os.path.join("file_manager", args.log_folder,
                                 "{}_{}.txt".format("file_manager_logs_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    logger.info("Script version is: {}".format(file_manager_args.version))

    pub = Publisher(args.db_path)
    directory_list = pub.get_directories_loc(args.recordings_dir, args.camera_no)
    pub.start_publishing(directory_list, args.cred_loc)
