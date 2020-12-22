CC = gcc
PYVERSION=3.8
FLAGS = -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python${PYVERSION} -o

default: compile clean

compile:
	cython mod_funcs.pyx
	${CC} ${FLAGS} mod_funcs.so mod_funcs.c

clean:
	rm mod_funcs.c

video:
	ffmpeg -r 12 -i ims/test%d.png -c:v libx264 -vf fps=12 -pix_fmt yuv420p out.mp4


