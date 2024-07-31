# Continuous Morph

## Setup

If it's your first time using a particular Pi, add your ssh key to GitHub, then:
- `ssh pi@faces2.local`
- `git clone git@github.com:camoverride/frankenface.git`
- `cd frankenface`
- `python -m venv --system-site-packages .venv`
- `source .venv/bin/activate`

Install [dlib](https://pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/):
- `cd ..`
- `git clone https://github.com/davisking/dlib.git`
- `cd dlib`
- `python setup.py install` (takes a long time)




Get the monitor dimensions:
- `fbset`

- `WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 180`
- `export DISPLAY=:0`

pip install numpy==1.4