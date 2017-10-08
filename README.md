# Darwin 2017

This is the code for our Darwin 2017 project on evolutionary algorithms.

## Code structure
The code is in two files: `solver.py` and `ea_util.pyx`. The former of these is the entry point for the program and the latter contains the code to solve the problems. This project implements code to solve four problems: OneMax, MaxSat, Ising spin glass (ISG), and the multidimensional knapsack problem (MKP).

## Cython
As Python isn't a particularly fast language, this project utilises [Cython](http://cython.org/) which enables python code to be compiled down to C. This makes the code run much faster but does mean that it doesn't play well with Windows operating systems. Changes to  `ea_util.pyx` won't be reflected in the running of the program without first compiling. This can be done by running the `make build` command from within the `darwin-2017` directory. This executes `setup.py` which compiles `ea_util.pyx` down to C code which can be called by Python in `solver.py`.

If a Windows operating system must be used, the `ea_util.pyx` file can be converted back to pure Python code by changing the file extension back to `.py` and removing the type declarations.

## Running the program
The program can be run by calling `python solver.py` with a number of arguments. Further explanation can be found at the top of the file or by executing `python solver.py -h`.

## Results
The program will save results to `results/$folder/$problem.csv` where `$folder` is a folder name created by the program based on the run parameters and `$problem.csv` is a CSV file containing a summary of each generation of the algorithm. In the case of the MKP, an extra directory layer is added between `results` and `$folder` and is named either `repair` or `no-repair` depending on whether the repair operator is used.

The package also provides a basic graph drawing utility which plots the average performance of a particular algorithm on a given problem. This isn't particularly well developed but does draw nice looking graphs.
