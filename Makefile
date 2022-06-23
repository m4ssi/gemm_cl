CC=gcc
CFLAGS=-Wall -g
LDFLAGS=-lOpenCL
EXE=gemm

all: $(EXE)

$(EXE): $(EXE).o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(EXE).o: $(EXE).c

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean: mrproper
	rm -rf *.o

mrproper:
	rm -rf $(EXE)
