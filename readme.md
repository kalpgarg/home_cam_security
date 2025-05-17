home_cam_security
A Python-based local surveillance solution that connects to DVR camera feeds, detects motion, and alerts you via Telegram.

🚀 Features
Implemented in Python

Leverages the simplicity and flexibility of Python for rapid development and deployment.

Multi-Camera Stream Fetching

Runs on a local server connected to a DVR.

Fetches streams from multiple cameras in parallel using the OpenCV module.

Motion Detection and Recording

Monitors each stream for motion in real-time.

On motion detection:

Starts recording from the specific camera.

Saves the recording locally.

Sends the recorded clip to a configured Telegram channel for instant alerts.

File and Log Management

Includes a set of utility scripts under the file_manager module to:

Automatically limit storage usage.

Regularly monitor and manage the logs/ directory to prevent uncontrolled growth.

📌 Requirements
Python 3.7+

OpenCV (cv2)

Telegram Bot API (with token and chat ID setup)

Other Python dependencies (see requirements.txt)

📲 Getting Started
Clone the repo:

'''
git clone https://github.com/your-username/home_cam_security.git
cd home_cam_security
'''
Install dependencies:
'''
pip install -r requirements.txt
Configure your Telegram bot and DVR stream details in the configuration file i.e. cam_info.json .
'''

Run the main server:

'''
python script_mon.py
'''
🔐 Security Note
Secure access to DVR and Telegram credentials is recommended.