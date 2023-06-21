import os

import zmq
from zmq.auth.thread import ThreadAuthenticator
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
import pathlib

base_path = pathlib.Path(__file__).parent.resolve()

class MessageList(BoxLayout):
    def __init__(self, **kwargs):
        super(MessageList, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.is_subscribed = False  # Flag to track subscription status

        # Create a start/stop button
        self.button = Button(text='Start', size_hint=(1, 0.1))
        self.button.bind(on_press=self.toggle_subscription)
        self.add_widget(self.button)

        # Create a recycle view
        self.recycle_view = RecycleView(data=[], viewclass='Label')
        self.add_widget(self.recycle_view)

    def toggle_subscription(self, *args):
        if self.is_subscribed:
            self.stop_subscription()
        else:
            self.start_subscription()

    def start_subscription(self):

        # Create a ZeroMQ subscriber
        context = zmq.Context()
        self.subscriber = context.socket(zmq.SUB)
        secret_keys_dir = os.path.join(base_path, "certificates", "private_keys")
        public_keys_dir = os.path.join(base_path, "certificates", "public_keys")
        # We need two certificates, one for the client and one for
        # the server. The client must know the server's public key
        # to make a CURVE connection.
        client_secret_file = os.path.join(secret_keys_dir, "subscriber.key_secret")
        client_public, client_secret = zmq.auth.load_certificate(client_secret_file)
        self.subscriber.curve_secretkey = client_secret
        self.subscriber.curve_publickey = client_public

        # The client must know the server's public key to make a CURVE connection.
        server_public_file = os.path.join(public_keys_dir, "publisher.key")
        server_public, _ = zmq.auth.load_certificate(server_public_file)
        self.subscriber.curve_serverkey = server_public

        self.subscriber.connect("tcp://localhost:9000")
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

        # Update button text and flag
        self.button.text = 'Stop'
        self.is_subscribed = True

        # Start receiving messages in a loop
        while self.is_subscribed:
            message = self.subscriber.recv_string()
            self.add_message(message)

    def stop_subscription(self):
        # Close the ZeroMQ subscriber
        self.subscriber.close()

        # Update button text and flag
        self.button.text = 'Start'
        self.is_subscribed = False

    def add_message(self, message):
        # Update the data in the recycle view
        self.recycle_view.data.append({'text': message})
        self.recycle_view.refresh_from_data()


class MyApp(App):
    def build(self):
        return MessageList()

def subscriber_without_app():
    print("123")
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    secret_keys_dir = os.path.join(base_path, "certificates", "private_keys")
    public_keys_dir = os.path.join(base_path, "certificates", "public_keys")
    # We need two certificates, one for the client and one for
    # the server. The client must know the server's public key
    # to make a CURVE connection.
    client_secret_file = os.path.join(secret_keys_dir, "subscriber.key_secret")
    client_public, client_secret = zmq.auth.load_certificate(client_secret_file)
    subscriber.curve_secretkey = client_secret
    subscriber.curve_publickey = client_public

    # The client must know the server's public key to make a CURVE connection.
    server_public_file = os.path.join(public_keys_dir, "publisher.key")
    server_public, _ = zmq.auth.load_certificate(server_public_file)
    subscriber.curve_serverkey = server_public

    subscriber.connect("tcp://localhost:9000")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    # Update button text and flag
    is_subscribed = True

    # Start receiving messages in a loop
    while is_subscribed:
        message = subscriber.recv_string()

        print(f"Received: {message}")

    # Close the subscriber
    subscriber.close()
    context.term()

if __name__ == '__main__':
    ignore_app = True
    if ignore_app:
        subscriber_without_app()
    else:
        MyApp().run()