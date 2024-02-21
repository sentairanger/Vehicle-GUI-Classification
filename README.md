# Vehicle-GUI-Classification
This project uses Guizero to display an image of a vehicle and show the color and vehicle type. To accomplish this, the project uses OpenVINO for vehicle detection.

## Getting Started

To get started, you will need a Camera Module (any will suffice), the Intel NCS2, and optionally a camera mount. Next, you will need to install OpenVINO on Raspberry Pi OS. This project has been tested on Bullseye but will be tried on Bookworm. This [tutorial](https://gist.github.com/sentairanger/caf11a2432ceebd715c6b33c224f4960) will help you with the process. Next, make sure to add the NCS2 and the Camera module. This will only work on the Desktop so it's best to use VNC and a tablet or phone to access the Pi remotely. Next, run the main application with `python3 vehicle-gui.py`. Then, press the `Take picture` button to take a picture. Then, detect the vehicle by pressing the `Detect Vehicle` button. Show the image with `Show result`. The image below shows a sample result.

![image](https://github.com/sentairanger/Vehicle-GUI-Classification/blob/main/car-gui.png)
