import zmq
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.listview import ListView, ListItemButton
from kivy.properties import ListProperty, ObjectProperty
from kivy.clock import Clock

class MessageList(ListView):
    messages = ListProperty([])

    def __init__(self, **kwargs):
        super(MessageList, self).__init__(**kwargs)

class MessageSubscriber(BoxLayout):
    message_list = ObjectProperty()

    def __init__(self, **kwargs):
        super(MessageSubscriber, self).__init__(**kwargs)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        # Set CURVE authentication
        self.socket.curve_secretkey = b'YOUR_CURVE_SECRET_KEY'
        self.socket.curve_publickey = b'YOUR_CURVE_PUBLIC_KEY'
        self.socket.curve_serverkey = b'YOUR_PUBLISHER_CURVE_PUBLIC_KEY'

        self.socket.connect("tcp://localhost:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')

    def start_listening(self):
        Clock.schedule_interval(self.receive_message, 0.1)

    def receive_message(self, dt):
        try:
            message = self.socket.recv_string(flags=zmq.NOBLOCK)
            self.message_list.messages.append(message)
        except zmq.ZMQError:
            pass

class SubscriberApp(App):
    def build(self):
        root = BoxLayout(orientation='vertical')
        message_list = MessageList()
        subscriber = MessageSubscriber(message_list=message_list)
        start_button = ListItemButton(text="Start Listening", on_release=subscriber.start_listening)
        root.add_widget(message_list)
        root.add_widget(start_button)
        return root

if __name__ == '__main__':
    SubscriberApp().run()