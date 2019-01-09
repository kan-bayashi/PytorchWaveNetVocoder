.PHONY: all clean

all: venv/bin/activate venv/lib/python3.6/site-packages/torch

venv/bin/activate:
	test -d venv || virtualenv -p python3.6 venv
	. venv/bin/activate; pip install -r requirements.txt; pip install matplotlib sprocket-vc;
	touch venv/bin/activate

venv/lib/python3.6/site-packages/torch: venv/bin/activate
	. venv/bin/activate; pip install pip --upgrade; pip install torch; pip install torchvision

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
