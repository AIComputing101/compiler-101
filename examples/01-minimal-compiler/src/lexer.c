/* src/lexer.c – Tokenizer
Converts input text into tokens.
*/
#include "token.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h> // Add this for exit()

// Helper: Check if next characters form "return"
static int is_return_keyword(FILE* input) {
    const char* keyword = "return";
    for (int i = 0; keyword[i] != '\0'; i++) {
        int c = fgetc(input);
        if (c != keyword[i]) {
            ungetc(c, input);
            return 0;
        }
    }
    int c = fgetc(input);
    if (isalnum(c) || c == '_') {
        ungetc(c, input);
        return 0;
    }
    ungetc(c, input);
    return 1;
}

// Get next token from input
Token next_token(FILE* input) {
    Token token;
    int c;

    while ((c = fgetc(input)) != EOF && isspace(c)) {}

    if (c == EOF) {
        token.type = TOKEN_EOF;
        return token;
    }

    if (c == 'r') {
        ungetc(c, input);
        if (is_return_keyword(input)) {
            token.type = TOKEN_RETURN;
            return token;
        }
    }

    if (isdigit(c)) {
        int value = 0;
        do {
            value = value * 10 + (c - '0');
            c = fgetc(input);
        } while (isdigit(c));
        ungetc(c, input);
        token.type = TOKEN_INTEGER;
        token.integer_value = value;
        return token;
    }

    if (c == ';') {
        token.type = TOKEN_SEMICOLON;
        return token;
    }

    fprintf(stderr, "Error: Unexpected character '%c'\n", c);
    exit(1); // Now recognized due to stdlib.h
}