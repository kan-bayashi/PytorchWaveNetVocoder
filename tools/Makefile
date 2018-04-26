.PHONY: all clean

all: venv/bin/activate venv/lib/python3.6/site-packages/sprocket venv/lib/python3.6/site-packages/torch

venv/bin/activate:
	test -d venv || virtualenv -p python3.6 venv
	. venv/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	touch venv/bin/activate

venv/lib/python3.6/site-packages/sprocket: venv/bin/activate
	git clone https://github.com/k2kobayashi/sprocket.git
	. venv/bin/activate; cd sprocket; pip install -r requirements.txt; python setup.py install

venv/lib/python3.6/site-packages/torch: venv/bin/activate
	. venv/bin/activate; pip install pip --upgrade; pip install torch; pip install torchvision

clean:
	rm -fr venv sprocket
	find -iname "*.pyc" -delete
