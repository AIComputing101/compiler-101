/* src/parser.c – Recursive Descent Parser
Validates token sequence and builds an AST.
*/
#include "parser.h"
#include "token.h" // For next_token() and Token
#include "ast.h"
#include <stdio.h>
#include <stdlib.h> // Add this for exit() and malloc()

static Token current_token;
static FILE* input_file;

static void next() {
    current_token = next_token(input_file); // Now recognized via token.h
}

static void expect(TokenType expected) {
    if (current_token.type != expected) {
        fprintf(stderr, "Error: Unexpected token\n");
        exit(1); // Now recognized via stdlib.h
    }
    next();
}

ReturnStmt* parse_return_stmt() {
    expect(TOKEN_RETURN);
    expect(TOKEN_INTEGER);
    ReturnStmt* stmt = malloc(sizeof(ReturnStmt)); // Now recognized via stdlib.h
    stmt->value = current_token.integer_value;
    expect(TOKEN_SEMICOLON);
    expect(TOKEN_EOF);
    return stmt;
}

ReturnStmt* parse(FILE* input) {
    input_file = input;
    current_token = next_token(input); // Now recognized via token.h
    return parse_return_stmt();
}