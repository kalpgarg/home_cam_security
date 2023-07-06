import psutil
import subprocess
import pathlib
import os

base_path = pathlib.Path(__file__).parent.resolve()

# List of script names to monitor and restart
script_names = ["fetch_save_stream.py", "file_manager.py"]

# Check if each script is running, and restart if not
for script_name in script_names:
    script_running = False
    for process in psutil.process_iter():
        if process.name() == "python":
            str_cmdline = " ".join(process.cmdline())
            index = str_cmdline.find(script_name)
            if index != -1:
                script_running = True
                break

    if not script_running:
        if script_name == script_names[0]:
            subprocess.Popen(["sudo", "sh", os.path.join(base_path, "sh_scripts", "fetch_streams.sh")])
            print(f"Restarted script: {script_name}")
        elif script_name == script_names[1]:
            subprocess.Popen(["sudo", "sh", os.path.join(base_path, "sh_scripts", "file_db_manager.sh")])
            print(f"Restarted script: {script_name}")
        else:
            print(f"Script name: {script_name} doesnt exist. Please check...")