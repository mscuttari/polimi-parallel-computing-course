CC = icc
CFLAGS = -O3 -lm

# Source file and target
SRC = Particles2D_2024.c
TARGET = Particles2D_2024

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

tar:
	tar -cf archive.tar stage*.ppm Population*.dmp Population.sta

# Clean rule
clean:
	rm -f $(TARGET) stage*.ppm normstage*.png Population*.dmp Population.sta
