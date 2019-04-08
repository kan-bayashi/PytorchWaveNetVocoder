.PHONY: all clean

all: venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv -p python3.6 venv
	. venv/bin/activate; pip install --upgrade pip
	. venv/bin/activate; pip install numpy
	. venv/bin/activate; pip install -r requirements.txt
	touch venv/bin/activate

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
