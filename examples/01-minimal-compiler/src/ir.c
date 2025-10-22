/* src/ir.c – IR Generator
Converts AST to TACKY IR.
*/
#include "ir.h"
#include "ast.h"
#include <stdlib.h> // Add this for malloc()

IR_Return* ast_to_ir(ReturnStmt* ast) {
    IR_Return* ir = malloc(sizeof(IR_Return)); // Now recognized
    ir->value = ast->value;
    return ir;
}