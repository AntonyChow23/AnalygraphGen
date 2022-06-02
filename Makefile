all: main
CC = g++
SRCS = main.cpp
PROG = main
CFLAGS = -std=c++11 $(shell pkg-config --cflags opencv4) -pthread
LIBS = $(shell pkg-config --libs opencv4)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) $(LIBS) $(SRCS) -o $(PROG) 

clean:
	rm -f *.o
	rm -rf main
