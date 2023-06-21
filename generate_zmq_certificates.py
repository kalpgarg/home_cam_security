#!/usr/bin/env python

"""
Generate publisher and subscriber CURVE certificate files then move them into the
appropriate store directory, private_keys or public_keys.

Author: Kalp Garg
"""

import os
import shutil
from typing import Union

import zmq.auth


def generate_certificates(base_dir: Union[str, os.PathLike]) -> None:
    '''Generate client and server CURVE certificate files'''
    keys_dir = os.path.join(base_dir, 'certificates')
    public_keys_dir = os.path.join(keys_dir, 'public_keys')
    secret_keys_dir = os.path.join(keys_dir, 'private_keys')

    # Create directories for certificates, remove old content if necessary
    for d in [keys_dir, public_keys_dir, secret_keys_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.mkdir(d)

    # create new keys in certificates dir
    server_public_file, server_secret_file = zmq.auth.create_certificates(
        keys_dir, "publisher"
    )
    client_public_file, client_secret_file = zmq.auth.create_certificates(
        keys_dir, "subscriber"
    )

    # move public keys to appropriate directory
    for key_file in os.listdir(keys_dir):
        if key_file.endswith(".key"):
            shutil.move(
                os.path.join(keys_dir, key_file), os.path.join(public_keys_dir, '.')
            )

    # move secret keys to appropriate directory
    for key_file in os.listdir(keys_dir):
        if key_file.endswith(".key_secret"):
            shutil.move(
                os.path.join(keys_dir, key_file), os.path.join(secret_keys_dir, '.')
            )


if __name__ == '__main__':
    if zmq.zmq_version_info() < (4, 0):
        raise RuntimeError(
            "Security is not supported in libzmq version < 4.0. libzmq version {}".format(
                zmq.zmq_version()
            )
        )

    generate_certificates(os.path.dirname(__file__))