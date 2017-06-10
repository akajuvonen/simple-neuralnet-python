all:	init

init:
	bin/init.sh

run:
	bin/init.sh
	bin/run.sh

test:
	nosetests -v

iris:
	PYTHONPATH=$(shell pwd) python tests/iris_test.py

plot:
	PYTHONPATH=$(shell pwd) python tools/sigmoid_plotter.py

clean:
	rm -fv *.pyc
	rm -fv tests/*.pyc
	rm -fv tools/*.pyc
	rm -rfv .env/
