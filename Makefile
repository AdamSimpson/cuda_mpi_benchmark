all: clean checkEnv ping_pong

.PHONY:        checkEnv clean

checkEnv:
ifndef CRAY_PRGENVGNU
	$(error Please load PrgEnv-gnu)
endif
ifndef CRAY_CUDATOOLKIT_VERSION
        $(error cudatoolkit module not loaded)
endif

ping_pong: pingpong.cpp
	mkdir -p bin
	CC -o bin/ping_pong pingpong.cpp	

clean:
	rm -f *.o
