/* src/ir.h – Intermediate Representation (TACKY IR) Definitions
Defines TACKY IR nodes.
*/
#ifndef IR_H
#define IR_H

#include "ast.h" // For ReturnStmt

// TACKY IR node for a return operation
typedef struct {
    int value; // Value to return (e.g., 42)
} IR_Return;

// Add prototype for ast_to_ir (defined in ir.c)
IR_Return* ast_to_ir(ReturnStmt* ast);

#endif // IR_H