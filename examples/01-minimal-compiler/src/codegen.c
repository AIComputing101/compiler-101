/* src/codegen.c – Code Generator
Converts TACKY IR to x64 assembly.
*/
#include "codegen.h"
#include <stdio.h>

void generate_asm(IR_Return* ir, FILE* output) {
    // macOS requires the text section to be explicitly marked
    fprintf(output, ".section __TEXT,__text\n");
    
    // On macOS, C functions are prefixed with '_' (mangling)
    fprintf(output, ".globl _main\n");  // Expose _main to the linker
    fprintf(output, "_main:\n");        // Entry point (with underscore)
    
    // x86_64 ABI: return value in %eax
    fprintf(output, "    movl $%d, %%eax\n", ir->value);
    
    // Return from _main
    fprintf(output, "    ret\n");
}