CC = gcc
PYVERSION=3.8
FLAGS = -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python${PYVERSION} -o

default: part1 part2 part3 clean

part1:
	cython agent_domain_2.pyx
	${CC} ${FLAGS} agent_domain_2.so agent_domain_2.c
      
part2:
	cython ccea_2.pyx
	${CC} ${FLAGS} ccea_2.so ccea_2.c
      
part3:
	cython reward_2.pyx
	${CC} ${FLAGS} reward_2.so reward_2.c

clean:
	rm *.c
	      
clear:
	rm *.c
	rm *.so
