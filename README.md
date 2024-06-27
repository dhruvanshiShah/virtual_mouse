# Computer Vision Based Virtual Mouse
- Designed a Python-based AI system using openCV for mouse control. Utilizing real-time camera input, the system detects hand landmarks and tracks gesture patterns, eliminating the need for a physical mouse.
- The index finger helps to track the pointer and mid-finger is used to select the point. The smoothness and frame margin can be controlled by updating the respective variables.

## Installation
- For cloning the Repository
````bash
git clone https://github.com/dhruvanshiShah/virtual_mouse.git
````
- For creating and activating new Conda environment
````bash
conda create --name py38 python=3.8
conda activate py38
````
- For installing the required packages
````bash
pip install -r requirements.txt
````
- For running the Script
````bash
python virtual_mouse.py
````
## Modules
_**hand_detection.py**_
- Detects the hand and tracks it <br>

_**virtual_mouse.py**_
- Main script to run virtual_mouse<br>

_**requirements.txt**_<br>
- Lists all the dependencies required for the project




  

