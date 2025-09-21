![Athena Walking](images/athena-walking-1.gif)

# Athena Robot Dog
### By [Matthew Thomas Beck](https://www.linkedin.com/in/matthewthomasbeck/)

Special thanks to [Aaed Musa](https://www.linkedin.com/in/aaedmusa/) and [Omar Ferrer](https://www.linkedin.com/in/omar-ferrer-0bb6355a/) for their help

Click [here](https://www.matthewthomasbeck.com/pages/athena.html) for the story behind the build

**Please consider:** if you like it, **star it!**

## Tech Stack
- **Language:** *Python*
- **Libraries:** *threading, queue, time, os, socket, logging, collections.deque, subprocess, signal, sys, random, numpy, opencv-python, openvino, smbus, RPi.GPIO, pigpio, pyserial*
- **Toolkits:** *OpenVino*

## Basic Information

Want to test/have fun with robot dogs, but don't want to spend multi-thousands for one? This project may be for you! Follow the instructions below to build one for yourself

## **(WIP)** Build For Yourself!
### Please carefully read every step before proceeding; robots get very expensive, very quickly

1. **(WIP)** Put together an Edge AI Module, instructions and part list found [here](https://github.com/matthewthomasbeck/Edge_AI_Module)

2. Asemble Aaed Musa's 'Ares' and buy hardware components like screws (**IMPORTANT!** do NOT buy the servos; instructions listed [here](https://www.instructables.com/3D-Printed-Robot-Dog/))

3. Purchase x12 45kg 8.4V servos: [AliExpress](https://www.aliexpress.us/item/3256808881971265.html?spm=a2g0o.productlist.main.3.6bd8OyjiOyjimW&algo_pvid=2d42210d-34c2-4142-b895-4343adc5442b&algo_exp_id=2d42210d-34c2-4142-b895-4343adc5442b-2&pdp_ext_f=%7B%22order%22%3A%2218%22%2C%22eval%22%3A%221%22%2C%22fromPage%22%3A%22search%22%7D&pdp_npi=6%40dis%21USD%2166.60%2122.14%21%21%21471.31%21156.64%21%402101effb17584197518685533ef113%2112000047801309564%21sea%21US%210%21ABX%211%210%21n_tag%3A-29910%3Bd%3A61b1bf00%3Bm03_new_user%3A-29895%3BpisId%3A5000000174221208&curPageLogUid=ddJlcghRXnfD&utparam-url=scene%3Asearch%7Cquery_from%3A%7Cx_object_id%3A1005009068286017%7C_p_origin_prod%3A)

4. Print out my custom parts and replace their respective Ares counterparts

5. Place the Edge AI Module into the robot; run the antenna through the antenna hole and wire the servos as follows:

![Alt Text](images/maestro.jpeg)

| **Joint Servo** | **Maestro Channel** |
| ----------- | ----------- |
| **Front Left Leg** |  |
| hip servo | *channel 3* |
| femur servo | *channel 5* |
| tibia servo | *channel 4* |
| **Front Right Leg** |  |
| hip servo | *channel 2* |
| femur servo | *channel 1* |
| tibia servo | *channel 0* |
| **Back Left Leg** |  |
| hip servo | *channel 9* |
| femur servo | *channel 7* |
| tibia servo | *channel 6* |
| **Back Right Leg** |  |
| hip servo | *channel 11* |
| femur servo | *channel 10* |
| tibia servo | *channel 9* |

5. **(WIP)** Wait for me to upload a comprehensive model (it takes time to train, on top of this being a first for me)