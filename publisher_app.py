"""
 *  @file  publisher_app.py
 *  @brief Look for new video stream addition and publishes it over rest endpoint
 *
 *  @author Kalp Garg.
"""
from datetime import datetime, timedelta
import pytz
import os
import time
import argparse
from waitress import serve
from py_logging import get_logger
# flask imports
from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
import uuid  # for public id
from werkzeug.security import generate_password_hash, check_password_hash
# imports for PyJWT authentication
import jwt
from datetime import datetime, timedelta
from functools import wraps
from common_utils import get_keys
import pathlib

base_path = pathlib.Path(__file__).parent.resolve()
global logger
global args

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


def create_database(app1):
    if not os.path.exists(os.path.join(base_path, 'user_db.db')):
        with app1.app_context():
            db.create_all()
            print("database created")

db = SQLAlchemy()
app = Flask(__name__)
# configuration
# NEVER HARDCODE YOUR CONFIGURATION IN YOUR CODE
# INSTEAD CREATE A .env FILE AND STORE IN IT
app.config['SECRET_KEY'] = get_keys(os.path.join(base_path, 'custom_cam_info.json'))
# database name
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_db.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# creates SQLALCHEMY object
db.init_app(app)

# Database ORMs
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
    last_index_fetched = db.Column(db.String(50))

User()
create_database(app)

# decorator for verifying the JWT
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        # return 401 if token is not passed
        if not token:
            return jsonify({'message': 'Token is missing !!'}), 401

        try:
            # decoding the payload to fetch the stored details
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query \
                .filter_by(public_id=data['public_id']) \
                .first()
        except:
            return jsonify({
                'message': 'Token is invalid !!'
            }), 401
        # returns the current logged in users context to the routes
        return f(current_user, *args, **kwargs)

    return decorated

@app.route('/')
def index():
    print("Route / has been called")
    server_ip = request.host.split(':')[0]
    return f"The server IP is: {server_ip}"

# User Database Route
# this route sends back list of users
@app.route('/user', methods=['GET'])
@token_required
def get_all_users(current_user):
    print("Route /user has been called")
    # querying the database
    # for all the entries in it
    users = User.query.all()
    # converting the query objects
    # to list of jsons
    output = []
    for user in users:
        # appending the user data json
        # to the response list
        output.append({
            'public_id': user.public_id,
            'name': user.name,
        })

    return jsonify({'users': output})


# route for logging user in
@app.route('/login', methods=['POST'])
def login():
    print("Route /login has been called")
    # creates dictionary of form data
    auth = request.form

    if not auth or not auth.get('name') or not auth.get('password'):
        # returns 401 if any name or / and password is missing
        return make_response(
            'Could not verify',
            401,
            {'WWW-Authenticate': 'Basic realm ="Login required !!"'}
        )

    user = User.query \
        .filter_by(name=auth.get('name')) \
        .first()

    if not user:
        # returns 401 if user does not exist
        return make_response(
            'Could not verify',
            401,
            {'WWW-Authenticate': 'Basic realm ="User does not exist !!"'}
        )

    if check_password_hash(user.password, auth.get('password')):
        # generates the JWT Token
        token = jwt.encode({
            'public_id': user.public_id,
            'exp': datetime.utcnow() + timedelta(minutes=30)
        }, app.config['SECRET_KEY'])

        return make_response(jsonify({'token': token.decode('UTF-8')}), 201)
    # returns 403 if password is wrong
    return make_response(
        'Could not verify',
        403,
        {'WWW-Authenticate': 'Basic realm ="Wrong Password !!"'}
    )


# signup route
@app.route('/signup', methods=['POST'])
def signup():
    print("Route /signup has been called")
    # creates a dictionary of the form data
    data = request.form

    # gets name and password
    name = data.get('username')
    password = data.get('password')

    # checking for existing user
    user = User.query \
        .filter_by(name=name) \
        .first()
    if not user:
        # database ORM object
        user = User(
            public_id=str(uuid.uuid4()),
            name=name,
            password=generate_password_hash(password)
        )
        # insert user
        db.session.add(user)
        db.session.commit()

        return make_response('Successfully registered.', 201)
    else:
        # returns 202 if user already exists
        return make_response('User already exists. Please Log in.', 202)

class Publisher(object):
    def __init__(self):
        pass

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

    def start_publishing(self, directory_list):
        processed_files = []
        for directory in directory_list:
            processed_files.append(set())
        while True:
            for i, directory in enumerate(directory_list):
                # Get the list of files in the directory
                files = os.listdir(directory)
                # Keep track of the files already processed
                new_files = set(files) - processed_files[i]
                # cntr = 0
                # Publish a message for each new file
                for file in new_files:
                    file_path = os.path.join(directory, file)
                    if file_path.endswith('.mp4'):
                        modified_time = os.path.getmtime(file_path)
                        current_time = time.time()
                        time_diff = current_time - modified_time

                        # If the file is new or modified within the last 1 seconds, publish it
                        logger.info("file_path: {}. Modified_time: {}".format(file_path, modified_time))
                        logger.info("Time diff is: {}".format(time_diff))
                        # if time_diff <= 60:
                        #     cntr += 1
                        #
                        #     if cntr < 2:
                        topic = directory  # Use the directory path as the topic
                        message = f"New file added: {file_path}"
                        self.socket.send_multipart([topic.encode(), message.encode()])
                        processed_files[i].add(file)
                        # cntr = 0

                for file in files:
                    file_path = os.path.join(directory, file)
                    if file_path.endswith('.mp4'):
                        modified_time = os.path.getmtime(file_path)
                        current_time = time.time()
                        time_diff = current_time - modified_time
                        if time_diff >= 1 * 24 * 60 * 60:  # if file is older than 2 days, delete it
                            try:
                                # Delete the file
                                os.remove(file_path)
                                logger.info(f"File '{file_path}' deleted successfully.")
                            except FileNotFoundError:
                                logger.error(f"File '{file_path}' not found.")
                            except PermissionError:
                                logger.error(f"Permission denied: unable to delete file '{file_path}'.")
                            except Exception as e:
                                logger.error(f"An error occurred while deleting the file: {str(e)}")


if __name__ == '__main__':
    publisher_args = argparse.ArgumentParser(description="Look for new video stream addition and publishes it "
                                             )
    publisher_args.version = "23.03.01"  # yy.mm.vv
    publisher_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    publisher_args.add_argument('-l', '--log_folder', type=str, metavar='zmq_publisher_log',
                                default="publisher_log",
                                help="Location of the log folder")
    publisher_args.add_argument('-cn', '--camera_no', action='store', type=list, default=[1],
                                metavar='123', help='Camera recording to publish. Default is 1. Range is 1 to 4')
    publisher_args.add_argument('-p', '--time_period', action='store', type=int, default=15,
                                metavar='10', help='Timeperiod of saving livestream. Default is 15')
    publisher_args.add_argument('-cl', '--config_file_loc', action='store', metavar='cam_info.json', type=str,
                                help='path of cam_info.json which contains user specific configurations', required=True)
    publisher_args.add_argument('-if', '--recordings_dir', action='store', metavar='cam_stream_log/recordings',
                                type=str,
                                help='path of recordings directory', required=True)

    args = publisher_args.parse_args()

    addl_file_loc = os.path.join("zmq_publisher", args.log_folder,
                                 "{}_{}.txt".format("zmq_publisher_logs_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    logger.info("Script version is: {}".format(publisher_args.version))


    # pub = Publisher(args.config_file_loc)
    # directory_list = pub.get_directories_loc(args.recordings_dir, args.camera_no)
    # pub.start_publishing(directory_list)
    dev_environ = False
    if dev_environ:
        app.run()
    else:
        serve(app, host="0.0.0.0", port=443)