TARGET = cn

SRC_DIR = src
OBJ_DIR = obj
CG_DIR  = $(SRC_DIR)/conjugate_gradient
CN_DIR  = $(SRC_DIR)/crank_nicolson
PGM_DIR = $(SRC_DIR)/pgm_utils
EXT_DIR = $(SRC_DIR)/external

SRCS = $(SRC_DIR)/main.c \
       $(CG_DIR)/conjugate_gradient.c \
       $(CN_DIR)/crank_nicolson.c \
       $(PGM_DIR)/pgm_utils.c

OBJS = $(OBJ_DIR)/main.o \
       $(OBJ_DIR)/conjugate_gradient.o \
       $(OBJ_DIR)/crank_nicolson.o \
       $(OBJ_DIR)/pgm_utils.o

CC = gcc

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    OPENCL_FLAGS = -framework OpenCL
else ifeq ($(UNAME_S),Linux)
    OPENCL_FLAGS = -lOpenCL
else ifeq ($(OS),Windows_NT)
    OPENCL_FLAGS = -lOpenCL
    CFLAGS += -I/mingw64/include
    CFLAGS += -L/mingw64/lib
    CFLAGS += -Wno-deprecated-declarations
endif


CFLAGS += -I$(SRC_DIR) -I$(CG_DIR) -I$(CN_DIR) -I$(PGM_DIR) -I$(EXT_DIR)


$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(OPENCL_FLAGS)


$(OBJ_DIR)/main.o: $(SRC_DIR)/main.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/conjugate_gradient.o: $(CG_DIR)/conjugate_gradient.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/crank_nicolson.o: $(CN_DIR)/crank_nicolson.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/pgm_utils.o: $(PGM_DIR)/pgm_utils.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
