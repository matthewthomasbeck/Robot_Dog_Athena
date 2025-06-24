[Service]
# Optional cleanup — don't fail if these pkill commands find nothing
ExecStartPre=-/usr/bin/pkill -f 'control_logic.py'
ExecStartPre=-/usr/bin/pkill -9 -f 'rpicam-jpeg|rpicam-vid|libcamera'

# Optional cleanup — don't fail if nothing is using serial0
ExecStartPre=-/bin/sh -c '/usr/bin/fuser -k /dev/serial0 || true'
ExecStopPost=-/bin/sh -c '/usr/bin/fuser -k /dev/serial0 || true'

ExecStart=
ExecStart=/home/matthewthomasbeck/.virtualenvs/openvino/bin/python /home/matthewthomasbeck/Projects/Robot_Dog/control_logic.py

Environment=VIRTUAL_ENV=/home/matthewthomasbeck/.virtualenvs/openvino
Environment=PYTHONPATH=/opt/intel/openvino/python/python3.9:/opt/intel/openvino/python/python3
Environment=LD_LIBRARY_PATH=/opt/intel/openvino/tools/compile_tool:/opt/intel/openvino/runtime/3rdparty/hddl/lib:/opt/intel/openvino/runtime/lib/aarch64
