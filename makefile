clean:
	rm -f -r build/
	rm -f ea_util.cpython-36m-x86_64-linux-gnu.so
	rm -f ea_util.c

clean-cpp:
	rm -f ea_utils.cpp

clean-all: clean clean-cpp

.PHONY: build
build: clean
	python setup.py build_ext --inplace

.PHONY: cython-build
cython-build: clean clean-cpp
	python setup.py build_ext --inplace --use-cython
