.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib -lcudart
CC	:= nvcc
DEP	:=  
ifeq ($(def), )
DEF := 
else
DEF := -D$(def)
endif

NVCCFLAGS	:= -lineinfo --ptxas-options=-v -g  

all:	race lock

race:	cmpe214_race.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o race cmpe214_race.o $(LIB)

lock:	cmpe214_lock.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o lock cmpe214_lock.o $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@
	

clean:
	rm -f *.o race lock


