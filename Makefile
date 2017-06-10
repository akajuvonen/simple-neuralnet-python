all:	init

init:
	bin/init.sh

run:
	bin/init.sh
	bin/run.sh

test:
	bin/init.sh
	bin/test.sh

iris:
	bin/init.sh
	bin/iris.sh

plot:
	PYTHONPATH=$(shell pwd) python tools/sigmoid_plotter.py

clean:
	rm -fv *.pyc
	rm -fv tests/*.pyc
	rm -fv tools/*.pyc
	rm -rfv .env/
