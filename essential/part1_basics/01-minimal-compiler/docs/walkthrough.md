# Walkthrough: Adding "return" Support

## 1. Lexer: Recognizing "return"  
- The lexer checks for the sequence 'r','e','t','u','r','n' using `is_return_keyword()`.  
- It ensures "return" is not part of a longer identifier (e.g., "returnXYZ" is rejected).  
- Example: Input "return" → `TOKEN_RETURN`.  


## 2. Parser: Validating "return <integer>;"  
- The parser expects:  
  1. `TOKEN_RETURN` (keyword)  
  2. `TOKEN_INTEGER` (value to return)  
  3. `TOKEN_SEMICOLON` (statement terminator)  
- If the sequence is invalid (e.g., missing semicolon), it throws an error.  


## 3. Code Generation: Platform-Specific Assembly  
To generate valid executable code, the code generator must account for OS differences:  

- **Linux**:  
  - Use `.text` to mark the executable section.  
  - Define `main` as the entry point (no underscore).  

- **macOS**:  
  - Use `.section __TEXT,__text` for the executable section.  
  - Prefix the entry point with `_` (e.g., `_main` instead of `main`) due to name mangling.  

Example (macOS):  
```asm
.section __TEXT,__text
.globl _main
_main:
    movl $42, %eax
    ret