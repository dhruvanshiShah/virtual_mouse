# virtual_mouse
Designed a Python-based AI system using openCV for mouse control. Utilizing real-time camera input, the system detects hand landmarks and tracks gesture patterns, eliminating the need for a physical mouse. The index finger helps to track the pointer and mid-finger is used to select the point. The smoothness and frame margin can be controlled by updating the respective variables.

#### Installation
````bash
git clone https://github.com/dhruvanshiShah/virtual_mouse.git
conda create --name py38 python=3.8
conda activate py38
pip install -r requirements.txt
python virtual_mouse.py
````

#### Installation
Inspired by the concepts presented in the tutorial, [https://www.youtube.com/watch?v=8gPONnGIPgw&t=397s]
