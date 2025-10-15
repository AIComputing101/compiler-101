#ifndef PARSER_H
#define PARSER_H

#include "ast.h"
#include <stdio.h>

// Parse input and return a ReturnStmt AST node
ReturnStmt* parse(FILE* input);

#endif // PARSER_H