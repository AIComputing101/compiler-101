/* src/driver.c – Orchestrates Compilation
Ties together lexing → parsing → IR → codegen, then links with GCC.
*/
#include "parser.h"
#include "ir.h"      // For ast_to_ir()
#include "codegen.h" // For generate_asm()
#include "ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input.txt>\n", argv[0]);
        return 1;
    }

    FILE* input = fopen(argv[1], "r");
    if (!input) { perror("fopen"); return 1; }

    ReturnStmt* ast = parse(input);
    fclose(input);

    IR_Return* ir = ast_to_ir(ast); // Now recognized via ir.h
    free(ast);

    FILE* asm_file = fopen("temp.s", "w");
    if (!asm_file) { perror("fopen"); return 1; }
    generate_asm(ir, asm_file); // Now recognized via codegen.h
    fclose(asm_file);
    free(ir);

    int exit_code = system("gcc temp.s -o output");
    if (exit_code != 0) {
        fprintf(stderr, "Linking failed\n");
        return 1;
    }

    printf("Compiled successfully to 'output'\n");
    return 0;
}