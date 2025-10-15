/* src/token.h – Shared Token Definitions
Defines token types used by the lexer and parser.
*/
#ifndef TOKEN_H
#define TOKEN_H

#include <stdio.h> // For FILE*

typedef enum {
    TOKEN_RETURN,   // "return" keyword
    TOKEN_INTEGER,  // Numeric literals (e.g., "42")
    TOKEN_SEMICOLON,// ";"
    TOKEN_EOF       // End of input
} TokenType;

typedef struct {
    TokenType type;
    int integer_value;  // For TOKEN_INTEGER (e.g., 42)
} Token;

// Add prototype for next_token (defined in lexer.c)
Token next_token(FILE* input);

#endif // TOKEN_H