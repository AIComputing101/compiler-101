/* src/ast.h – Abstract Syntax Tree (AST) Definitions
Defines the structure of parsed code.
*/
#ifndef AST_H
#define AST_H

// AST node for a return statement: "return <integer>;"
typedef struct {
    int value; // The integer to return (e.g., 42)
} ReturnStmt;

#endif // AST_H