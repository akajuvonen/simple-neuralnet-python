init:
	pip install -r requirements.txt

run:
	python neuralnet.py

test:
	nosetests -v

iris:
	PYTHONPATH=$(shell pwd) python tests/iris_test.py

clean:
	rm -fv *.pyc
	rm -fv tests/*.pyc
	rm -fv tools/*.pyc
