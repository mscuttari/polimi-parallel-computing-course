NVCC = nvcc
CFLAGS = -O3 -arch sm_80 -lmpi -Xcompiler=-fopenmp

# Source file and target
SRC = Particles2D_2024.cu
TARGET = Particles2D_2024

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

tar:
	tar -cf archive.tar stage*.ppm Population*.dmp Population.sta

# Clean rule
clean:
	rm -f $(TARGET) stage*.ppm normstage*.png Population*.dmp Population.sta
