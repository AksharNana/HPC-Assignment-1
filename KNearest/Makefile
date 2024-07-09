# Compiler
CC = gcc
# Compiler flags
CFLAGS = -fopenmp -O3
# Source files
SOURCES = main.c \
          src/qsort.c \
          src/bfKNN.c

# Header files
HEADERS = headers/qsort.h \
          headers/bfKNN.h \
          headers/type.h
# Libraries
LIBS = -lm
# Target executable
TARGET = KNearest

TASK_CFLAGS = -DPARALLEL -DTASK 
SECTION_CFLAGS = -DPARALLEL -DSECTION
SERIAL_CFLAGS = -DSERIAL

# Default target
all: serial

serial: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LIBS) $(SERIAL_CFLAGS)

task: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LIBS) $(TASK_CFLAGS)

section: $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LIBS) $(SECTION_CFLAGS)
        
# Clean intermediate and executable files
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
