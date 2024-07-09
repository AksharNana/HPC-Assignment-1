INC="./inc"
INCFLAGS=-I$(INC)
CC=gcc
NVCC=nvcc

all: imageConv

imageConv: imageConv.cu
	$(NVCC) $(INCFLAGS) imageConv.cu -o imageConv

clean:
	rm imageConv