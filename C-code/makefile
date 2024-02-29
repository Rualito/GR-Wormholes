# Makefile is in main folder
CC:= g++
NVCC:= /opt/cuda/bin/nvcc

VPATH = src/:lib/ 
CFLAGS:= -O3 -m64 -std=c++11 -g -use_fast_math

CUDA_PATH ?= /opt/cuda
CUDA_PATH := -I$(CUDA_PATH)
CUDALIB   := -I/opt/cuda/samples/common/inc/
CUDAFLAGS:= -gencode arch=compute_61,code=sm_61 -ccbin $(CC) -m64

SFMLLIB := -lsfml-graphics -lsfml-window -lsfml-system -lsfml-network -lsfml-audio 

ROOTINC:= $(shell root-config --cflags)

ROOTLIB:= $(shell root-config --libs)

SRC:= $(notdir $(wildcard src/*.cpp)) $(notdir $(wildcard src/*.C)) $(notdir $(wildcard src/*.cu))

LIB:= $(notdir $(wildcard lib/*.cpp)) $(notdir $(wildcard lib/*.C)) $(notdir $(wildcard lib/*.cu))

SRC_cpp :=$(filter %.cpp, $(SRC))
SRC_C   :=$(filter %.C, $(SRC))
SRC_cu  :=$(filter %.cu, $(SRC))

LIB_cpp := $(filter %.cpp, $(LIB))
LIB_C   :=$(filter %.C, $(LIB))
LIB_cu  := $(filter %.cu, $(LIB))

SRC_OBJ := $(addprefix bin/, $(SRC_cpp:.cpp=.o))
SRC_OBJ += $(addprefix bin/, $(SRC_C:.C=.o))

SRC_OBJ += $(addprefix bin/, $(SRC_cu:.cu=.o))


LIB_OBJ := $(addprefix bin/, $(LIB_cpp:.cpp=.o))
LIB_OBJ += $(addprefix bin/, $(LIB_C:.C=.o)) 

LIB_OBJ_CU := $(addprefix bin/, $(LIB_cu:.cu=.o)) 

OBJS:= $(SRC_OBJ) $(LIB_OBJ)

CUDA_OBJ := $(SRC_OBJ_cu) $(LIB_OBJ_cu)

EXE:= $(addprefix bin/, $(MAIN:.cpp=.x))

INCLUDE := -I src -I lib

CUDA_COMPILE := 0

all : $(EXE)

lib: $(LIB_OBJ)

src: $(SRC_OBJ)

obj: $(OBJS)

cuobj: $(CUDA_OBJ)

bin/%.x : bin/%.o $(OBJS)
	@echo "Making executable $< [$@]"
	-$(NVCC)  $(CUDAFLAGS) -o $@ $^ $(INCLUDE) -I$(CUDA_PATH) $(CUDA_LIB) $(SFMLLIB) 
	@echo 
#@echo "Running executable"
#$@


#necessary for extension .cpp
bin/%.o : %.cpp 
	@echo "Compiling $< [$@]"
	@$(CC) $(CFLAGS) -c -o $@ $< $(INCLUDE) $(ROOTINC)  #$(#SFMLLIB)

# necessary for extension .cu # )
bin/%.o : %.cu
	@echo "Compiling $< [$@]"
	@-$(NVCC)  $(CUDAFLAGS) -c -o $@ $< $(INCLUDE) $(CUDA_PATH) $(CUDA_LIB) $(LIBDEVICE) $(SFMLLIB)

#needed for extension .C
bin/%.o : %.C
	@echo "Compiling $< [$@]"
	@$(CC) $(CFLAGS) -c -o $@ $< $(INCLUDE) $(ROOTINC)


clean:
	@rm -fv $(LIB_OBJ) $(SRC_OBJ) $(EXE) $(CUDA_OBJ)

clnlib:
	@rm -fv $(LIB_OBJ)

clnsrc:
	@rm -fv $(SRC_OBJ)

clnexe:
	@rm -fv $(EXE)

shwroot:
	@echo "ROOT INCLUDE"
	@echo $(ROOTINC)

shwsrc:
	@echo "SOURCE .cpp"
	@echo $(SRC_OBJ)
	@echo

shwlib:
	@echo "LIBRARY .cpp"
	@echo $(LIB)
	@echo
shwcuda:
	@echo "CUDA .cu"
	@echo $(SRC_cu) $(LIB_cu)

shwall:
	@echo "ROOT INCLUDE"
	@echo $(ROOTINC)
	@echo 

	@echo "SOURCE .cpp"
	@echo $(SRC)
	@echo  

	@echo "LIB .cpp"
	@echo $(LIB)
	@echo

