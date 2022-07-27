.PHONY: all clean

all: a.out

a.out : ex5_10.cpp ex5_10.h
	clang++ -std=c++11 -o $@ $<

ex5_10 : ex5_10.cpp ex5_10.h
	clang++ -std=c++11 -DNDEBUG -O3 -o $@ $<

clean:
	rm -f ex5_10
	rm -f a.out
