# home_cam_security

A Python-based local surveillance solution that connects to DVR camera feeds, detects motion, and alerts you via Telegram.

## 🚀 Features

- **Implemented in Python**
  - Built using the power and flexibility of Python.

- **Multi-Camera Stream Fetching**
  - Runs on a server connected to a DVR.
  - Fetches streams from multiple cameras **in parallel** using the `opencv-python` (`cv2`) module.

- **Motion Detection and Recording**
  - Monitors each stream for motion in real-time.
  - On detecting motion:
    - Starts recording from the specific camera.
    - Saves the recording locally.
    - Sends the recorded clip to a **Telegram channel** for instant alerts.

- **File and Log Management**
  - Includes utility scripts under the `file_manager` directory:
    - Manages disk space by automatically removing old recordings.
    - Monitors the `logs/` folder to prevent uncontrolled growth.

## 📲 Getting Started

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/your-username/home_cam_security.git
cd home_cam_security
pip install -r requirements.txt
```

### 2. Configure your environment
- Set up DVR stream URLs.
- Add your Telegram Bot API key and channel ID.
- Use custom_cam_info.json to store your settings securely.


### 3. Run the main server
```bash
python script_mon.py
```

## 🔐 Security Note
- Make sure to secure your DVR access credentials and Telegram bot tokens appropriately.

## 📬 Contributions
- Contributions are welcome!
- Feel free to fork the repo and submit a pull request for features, bug fixes, or improvements.


