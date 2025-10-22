#ifndef CODEGEN_H
#define CODEGEN_H

#include "ir.h" // For IR_Return
#include <stdio.h> // For FILE*

// Add prototype for generate_asm (defined in codegen.c)
void generate_asm(IR_Return* ir, FILE* output);

#endif // CODEGEN_H