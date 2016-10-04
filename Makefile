init:
	pip install -r requirements.txt

run:
	python neuralnet.py

test:
	nosetests

iris:
	PYTHONPATH=$(shell pwd) python tests/iris_test.py

clean:
	rm -v *.pyc
	rm -v tests/*.pyc
	rm -v tools/*.pyc
