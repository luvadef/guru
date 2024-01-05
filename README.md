To run the project, you must install Python, Miniconda3 and Tensorflow:

Python download: https://www.python.org/downloads/
Miniconda3 download: https://docs.conda.io/en/latest/miniconda.html

Install the TensorFlow dependencies:
conda install -c apple tensorflow-deps

Install base TensorFlow:
pip install tensorflow-macos

Install Metal plugin:
pip install tensorflow-metal

Now install common additional packages and upgrade the packages so that they are updated to the M1 architecture.

pip install numpy  --upgrade
pip install pandas  --upgrade
pip install matplotlib  --upgrade
pip install scikit-learn  --upgrade
pip install scipy  --upgrade
pip install plotly  --upgrade
