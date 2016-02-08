CC = icc
CXX = icpc

ARCH = ar
ARCHFLAGS = cr
RANLIB = ranlib

ifeq ($(BLISLAB_USE_BLAS),true)
CFLAGS = -O3 -fopenmp -mavx -fPIC -DUSE_BLAS
else
CFLAGS = -O3 -fopenmp -mavx -fPIC
endif

LIBBLISLAB = $(BLISLAB_DIR)/lib/libblisgemm.a
SHAREDLIBBLISLAB = $(BLISLAB_DIR)/lib/libblisgemm.so

LDFLAGS = -I$(BLISLAB_DIR)/include -I$(BLISLAB_DIR)/kernels -I/$(BLISLAB_MKL_DIR)/include

LDLIBS = $(LIBBLISLAB) -lpthread -lm -openmp -mkl=parallel -Werror -Wall -pedantic
