import psutil
import subprocess
import pathlib
import os
import time

base_path = pathlib.Path(__file__).parent.resolve()

# Script names to monitor
script_names = ["fetch_save_stream.py", "file_manager.py"]

# Define expected camera numbers for fetch_save_stream.py
expected_cameras = [1, 2, 3, 4]

while True:
    # Track running cameras for fetch_save_stream.py
    running_cameras = set()

    # Track if file_manager.py script is running
    file_manager_running = False

    for process in psutil.process_iter():
        try:
            if process.name() == "python":
                cmdline = process.cmdline()
                str_cmdline = " ".join(cmdline)

                # Check for fetch_save_stream.py running cameras
                if "fetch_save_stream.py" in str_cmdline:
                    # Find the camera number after '-cn'
                    if "-cn" in cmdline:
                        cn_index = cmdline.index("-cn")
                        if cn_index + 1 < len(cmdline):
                            cam_num_str = cmdline[cn_index + 1]
                            try:
                                cam_num = int(cam_num_str)
                                running_cameras.add(cam_num)
                            except ValueError:
                                print(f"Warning: Unable to parse camera number from '-cn {cam_num_str}'")
                
                # Check if file_manager.py is running
                if "file_manager.py" in str_cmdline:
                    file_manager_running = True

        except Exception as e:
            print("Exception occurred while checking processes:", e)

    # Restart fetch_save_stream.py instances for cameras not running
    for cam in expected_cameras:
        if cam not in running_cameras:
            print(f"Camera {cam} for fetch_save_stream.py is not running, restarting...")
            script_path = os.path.join(base_path, "sh_scripts", "fetch_streams.sh")
            # Assuming the shell script can take camera number as argument to start specific instance
            subprocess.Popen([script_path, str(cam)])
            print(f"Restarted fetch_save_stream.py for camera {cam}")

    # Restart file_manager.py if not running
    if not file_manager_running:
        script_path = os.path.join(base_path, "sh_scripts", "file_db_manager.sh")
        subprocess.Popen([script_path])
        print("Restarted script: file_manager.py")

    time.sleep(15)

