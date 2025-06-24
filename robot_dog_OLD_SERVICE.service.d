### Editing /etc/systemd/system/robot_dog.service.d/override.conf
### Anything between here and the comment below will become the new contents of the file

[Service]
# Explicit Python interpreter inside virtualenv
ExecStart=
ExecStart=/home/matthewthomasbeck/.virtualenvs/openvino/bin/python /home/matthewthomasbeck/Projects/Robot_Dog/control_logic.py

# Add required environment variables for OpenVINO + virtualenv
Environment=VIRTUAL_ENV=/home/matthewthomasbeck/.virtualenvs/openvino
Environment=PYTHONPATH=/opt/intel/openvino/python/python3.9:/opt/intel/openvino/python/python3
Environment=LD_LIBRARY_PATH=/opt/intel/openvino/tools/compile_tool:/opt/intel/openvino/runtime/3rdparty/hddl/lib:/opt/intel/openvino/runtime/lib/aarch64

### Lines below this comment will be discarded

### /etc/systemd/system/robot_dog.service
# [Unit]
# Description=Control Logic Service
# After=network.target
#
# [Service]
# ExecStartPre=/bin/sleep 10
# ExecStart=/bin/bash -c 'source /home/matthewthomasbeck/.virtualenvs/openvino/bin/activate && python /home/matthewthomasbeck/Projects/Robot_Dog/control_logic.py'
# WorkingDirectory=/home/matthewthomasbeck/Projects/Robot_Dog
# StandardOutput=inherit
# StandardError=inherit
# Restart=on-abnormal
# User=matthewthomasbeck
# Environment="DISPLAY=:0" "XAUTHORITY=/home/matthewthomasbeck/.Xauthority" "XDG_RUNTIME_DIR=/run/user/1000"
# TimeoutStartSec=600
#
# [Install]
# WantedBy=multi-user.target