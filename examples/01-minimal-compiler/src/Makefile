CC = gcc
CFLAGS = -Wall -Wextra

# Target: minimal-compiler (the driver)
minimal-compiler: driver.c lexer.c parser.c ir.c codegen.c
	$(CC) $(CFLAGS) -o minimal-compiler $^

clean:
	rm -f minimal-compiler temp.s output