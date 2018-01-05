.PHONY: all clean

all: venv/bin/activate venv/lib/python2.7/site-packages/torch

# for some reason, including matplotlib in requirements.txt causes errors, and
# matplotlib should be separately pip installed
venv/bin/activate:
	test -d venv || virtualenv -p /usr/bin/python2.7 venv
	. venv/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	touch venv/bin/activate


venv/lib/python2.7/site-packages/torch: venv/bin/activate
	. venv/bin/activate; pip install pip --upgrade; pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl; pip install torchvision

clean:
	rm -fr venv 
	find -iname "*.pyc" -delete
