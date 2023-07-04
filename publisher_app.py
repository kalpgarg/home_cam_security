"""
 *  @file  publisher_app.py
 *  @brief Server for user creation and publishes video it over REST endpoint
 *
 *  @author Kalp Garg.
"""
import os
import argparse
from waitress import serve
from py_logging import get_logger
# flask imports
from flask import Flask, request, jsonify, make_response, send_file
from flask_sqlalchemy import SQLAlchemy
import uuid  # for public id
from werkzeug.security import generate_password_hash, check_password_hash
# imports for PyJWT authentication
import jwt
from datetime import datetime, timedelta
from functools import wraps
from common_utils import get_keys, return_datetime
import pathlib

base_path = pathlib.Path(__file__).parent.resolve()
global logger


def create_database(app1):
    if not os.path.exists(os.path.join(base_path, 'user_db.db')):
        with app1.app_context():
            db.create_all()
            print("database created")


db = SQLAlchemy()
app = Flask(__name__)
# configuration
# NEVER HARDCODE YOUR CONFIGURATION IN YOUR CODE
app.config['SECRET_KEY'] = get_keys(os.path.join(base_path, 'custom_cam_info.json'))
# database name
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_db.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# creates SQLALCHEMY object
db.init_app(app)


# Database ORMs
class Recordings(db.Model):
    __tablename__ = 'recordings'
    id = db.Column(db.Integer, primary_key=True)
    index_record = db.Column(db.Integer, unique=True)
    cam_no = db.Column(db.Integer)
    file_path = db.Column(db.String(200), unique=True)
    cam_loc = db.Column(db.String(20))
    from_dnt = db.Column(db.DateTime(timezone=True))
    to_dnt = db.Column(db.DateTime(timezone=True))
    # user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
    user_created_at = db.Column(db.DateTime(timezone=True), default=datetime.now)
    last_index_fetched = db.Column(db.String(50))
    # recordings = db.relationship('Recordings')


User()
Recordings()
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
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query \
                .filter_by(public_id=data['public_id']) \
                .first()
        except Exception as e:
            print(e)
            return jsonify({
                'message': 'Token is invalid !!'
            }), 401
        # returns the current logged in users context to the routes
        return f(current_user, *args, **kwargs)

    return decorated


# User Database Route
# this route sends back list of users
@app.route('/user', methods=['GET'])
@token_required
def get_all_users(current_user):
    logger.info("Route /user has been called")
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


@app.route('/new-data/<int:last_fetched>', methods=['GET'])
@token_required
def get_new_data(current_user, last_fetched):
    logger.info("Route /new-data has been called")
    if last_fetched is None:
        logger.info("Requires last fetched index.")
        return make_response(
            'Could not fetch',
            401,
            {'WWW-Authenticate': 'Last fetched index is required'}
        )
    logger.info("last fetched index: {}".format(last_fetched))
    # querying the database
    # for all the entries in it
    new_data = Recordings.query.filter(Recordings.index_record > last_fetched)
    output = []
    for data in new_data:
        # appending the user data json
        # to the response list
        output.append({
            'index': data.index_record,
            'cam_no': data.cam_no,
            'file_path': data.file_path,
            'cam_loc': data.cam_loc,
            'from_dnt': data.from_dnt,
            'to_dnt': data.to_dnt
        })
    last_index = Recordings.query.order_by(Recordings.index_record.desc()).first()
    current_user.last_index_fetched = int(last_index.index_record)
    logger.info("Last index updated to {} for user {}".format(last_index, current_user))
    db.session.commit()

    return jsonify({'new_data': output}), 200

@app.route('/fetch-data/<int:file_index>', methods=['GET'])
@token_required
def get_video(current_user, file_index):
    logger.info("Route /fetch-data has been called")
    if file_index is None:
        logger.info("Requires index of file.")
        return make_response(
            'Could not fetch',
            401,
            {'WWW-Authenticate': 'File index is required'}
        )
    logger.info("File index: ", file_index)
    # querying the database
    # for all the entries in it
    new_data = Recordings.query.filter_by(index_record=file_index)
    video_fpath = new_data.file_path
    if not os.path.exists(video_fpath):
        logger.info(f"Video file path {video_fpath} doesn't exist..")
        return make_response(
            'File path not exist',
            403,
            {'WWW-Authenticate': 'File path not exist'}
        )
    return send_file(video_fpath, as_attachment=True), 200

# route for logging user in
@app.route('/login', methods=['POST'])
def login():
    logger.info("Route /login has been called")
    # creates dictionary of form data
    auth = request.form
    if not auth or not auth.get('username') or not auth.get('password'):
        # returns 401 if any name or / and password is missing
        return make_response(
            'Could not verify',
            401,
            {'WWW-Authenticate': 'Basic realm ="Login required !!"'}
        )

    user = User.query \
        .filter_by(name=auth.get('username')) \
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
            'exp': datetime.utcnow() + timedelta(minutes=60),
            'last_index': user.last_index_fetched
        }, app.config['SECRET_KEY'], algorithm='HS256')
        return make_response(jsonify({'token': token}), 200)
    # returns 403 if password is wrong
    return make_response(
        'Could not verify',
        403,
        {'WWW-Authenticate': 'Basic realm ="Wrong Password !!"'}
    )


# signup route
@app.route('/signup', methods=['POST'])
def signup():
    logger.info("Route /signup has been called")
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
            password=generate_password_hash(password),
            last_index_fetched=1,
            user_created_at=datetime.utcnow()
        )
        # insert user
        db.session.add(user)
        db.session.commit()

        return make_response('Successfully registered.', 201)
    else:
        # returns 202 if user already exists
        return make_response('User already exists. Please Log in.', 409)


if __name__ == '__main__':
    publisher_args = argparse.ArgumentParser(description="server for creating a user and sending data"
                                             )
    publisher_args.version = "23.03.01"  # yy.mm.vv
    publisher_args.add_argument('-v', '--version', action='version', help="displays the version. Format = yy.mm.v")
    publisher_args.add_argument('-l', '--log_folder', type=str, metavar='publisher_log',
                                default="publisher_log",
                                help="Location of the log folder")
    args = publisher_args.parse_args()

    addl_file_loc = os.path.join("publisher", args.log_folder,
                                 "{}_{}.txt".format("publisher_logs_", return_datetime(mode=1)))
    logger = get_logger(__name__, addl_file_loc, save_to_file=True)
    logger.info("Script version is: {}".format(publisher_args.version))

    dev_environ = False
    if dev_environ:
        app.run()
    else:
        serve(app, host="0.0.0.0", port=443)
