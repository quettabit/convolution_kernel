TARGET_1 := cudnn_convolution
TARGET_2 := convolution
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAG_1 := -arch=sm_35
CXXFLAG_2 := -std=c++11 -O2
CXXFLAG_3 := -Wno-deprecated-gpu-targets

all: conv1 conv2

conv1: $(TARGET_1).cu
        $(CXX) $(CXXFLAG_1) $(CXXFLAG_2) $(HEADERS) $(LIBS) $(TARGET_1).cu -o $(TARGET_1) \
        -lcudnn
conv2: $(TARGET_2).cu
        $(CXX) $(CXXFLAG_3) $(TARGET_2).cu -o $(TARGET_2)

.phony: clean

clean:
        rm -f $(TARGET_1) $(TARGET_2) || echo -n "" 
