"""
 *  @file  publisher_app.py
 *  @brief Look for new video stream addition and publishes it over zmq messaging
 *
 *  @author Kalp Garg.
"""
from datetime import datetime, timedelta
import pytz
import os
import time
import zmq
import argparse
from py_logging import get_logger
from zmq.auth.thread import ThreadAuthenticator

global logger

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


class ZmqPublisher(object):
    def __init__(self, keys_dir):

        public_keys_dir = os.path.join(keys_dir, 'public_keys')
        secret_keys_dir = os.path.join(keys_dir, 'private_keys')

        if not (
                os.path.exists(keys_dir)
                and os.path.exists(public_keys_dir)
                and os.path.exists(secret_keys_dir)
        ):
            logger.error(
                "Certificates are missing: run generate_certificates.py script first. Quitting..."
            )
            quit()
        # Set up ZeroMQ context and socket
        context = zmq.Context()


        # Start an authenticator for this context.
        auth = ThreadAuthenticator(context)
        auth.start()
        auth.allow('127.0.0.1')
        # Tell the authenticator how to handle CURVE requests
        auth.configure_curve(domain='*', location=zmq.auth.CURVE_ALLOW_ANY)

        self.socket = context.socket(zmq.PUB)
        pub_secret_file = os.path.join(secret_keys_dir, "publisher.key_secret")
        server_public, server_secret = zmq.auth.load_certificate(pub_secret_file)
        self.socket.curve_secretkey = server_secret
        self.socket.curve_publickey = server_public
        self.socket.curve_server = True  # must come before bind
        self.socket.bind('tcp://*:9000')

    def get_directories_loc(self, main_recording_dir, cam_list):
        directories= []

        for cam_no in cam_list:
            cam_recording_path = os.path.join(main_recording_dir, "cam{}".format(cam_no))
            if not os.path.exists(cam_recording_path):
                logger.error(
                    "Given recordings directory {} doesn't exist.Please check.. Quitting...".format(cam_recording_path))
                quit()
            directories.append(cam_recording_path)
        logger.info("Directories to look: {}".format(directories))
        return directories
    def start_publishing(self, directory_list):

        while True:
            for directory in directory_list:

                # Get the list of files in the directory
                files = os.listdir(directory)

                # Publish a message for each new file
                for file in files:
                    file_path = os.path.join(directory, file)
                    modified_time = os.path.getmtime(file_path)
                    # logger.info("file_path: {}. Modified_time: {}".format(file_path, modified_time))
                    current_time = time.time()
                    time_diff = current_time - modified_time
                    # logger.info("Time diff is: {}".format(time_diff))
                    # If the file is new or modified within the last 1 seconds, publish it
                    if time_diff <= 1:
                        topic = directory  # Use the directory path as the topic
                        message = f"New file added: {file_path}"
                        self.socket.send_multipart([topic.encode(), message.encode()])

if __name__ == '__main__':
    zmq_publisher_args = argparse.ArgumentParser(description="Look for new video stream addition and publishes it "
                                                             "over zmq messaging")
    zmq_publisher_args.version = "23.03.01"  # yy.mm.vv
    zmq_publisher_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    zmq_publisher_args.add_argument('-l', '--log_folder', type=str, metavar='zmq_publisher_log',
                                    default="zmq_publisher_log",
                                    help="Location of the log folder")
    zmq_publisher_args.add_argument('-cn', '--camera_no', action='store', type=list, default=[1],
                                    metavar='123', help='Camera recording to publish. Default is 1. Range is 1 to 4')
    zmq_publisher_args.add_argument('-p', '--time_period', action='store', type=int, default=15,
                                    metavar='10', help='Timeperiod of saving livestream. Default is 15')
    zmq_publisher_args.add_argument('-kd', '--keys_dir', action='store', metavar='certificates', type=str,
                                    help='path of keys dir. Generate using generate_zmq_certificates.py', required=True)
    zmq_publisher_args.add_argument('-if', '--recordings_dir', action='store', metavar='cam_stream_log/recordings', type=str,
                                    help='path of recordings directory', required=True)

    args = zmq_publisher_args.parse_args()

    addl_file_loc = os.path.join("zmq_publisher", args.log_folder,
                                 "{}_{}.txt".format("zmq_publisher_logs_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    logger.info("Script version is: {}".format(zmq_publisher_args.version))


    zmq_pub = ZmqPublisher(args.keys_dir)
    directory_list = zmq_pub.get_directories_loc(args.recordings_dir, args.camera_no)
    zmq_pub.start_publishing(directory_list)
