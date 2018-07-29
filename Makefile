.PHONY: all clean

all:  KNNFloatImg compareFloatImg floatImage2png test
	
incs=-I /home/include/png++/0.2.5_1/include 

floatImage2png: floatImage2png.cpp
	g++ -O3 -g -std=c++11 $(incs) -o floatImage2png floatImage2png.cpp `libpng-config --cflags` `libpng-config --ldflags`

KNNFloatImg: KNNFloatImg.cpp
	g++ -O3 -g -std=c++11 $(incs) -o KNNFloatImg KNNFloatImg.cpp -fopenmp

compareFloatImg: compareFloatImg.cpp
	g++ -O3 -g -std=c++11 $(incs) -o compareFloatImg compareFloatImg.cpp

test: test.cpp
	g++ -O3 -g -std=c++11 test.cpp -o test
	
clean:	
	rm -rf floatImage2png floatImage2png *.dSYM

