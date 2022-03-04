all: libdtw.so

dtw.o: src/dtw.c
	gcc -Wall -fPIC -c src/dtw.c

libdtw.so: dtw.o
	gcc -Wall -shared -o libdtw.so dtw.o

clean:
	rm -f *.o *.so