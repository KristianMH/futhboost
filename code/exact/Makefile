all: bench

bench:
	futhark bench exact-clas.fut --backend=cuda

clean:
	find . -maxdepth 1 -type f -executable -exec rm '{}' \;
	rm *.c

