.PHONY: all clean

all: venv/bin/activate

venv/bin/activate:
	test -d venv || virtualenv -p python3.6 venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && cd ../ &&  pip install torch==1.0.1 torchvision==0.2.2
	. venv/bin/activate && cd ../ &&  pip install -e .
	. venv/bin/activate && cd ../ &&  pip install -e .[test]

clean:
	rm -fr venv
	find -iname "*.pyc" -delete
