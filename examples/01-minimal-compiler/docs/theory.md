# Theory: 4 Core Compiler Passes

This module implements 4 fundamental compiler stages:

1. **Lexical Analysis (Lexer)**  
   Converts raw text into a stream of tokens (e.g., "return 42;" → `TOKEN_RETURN`, `TOKEN_INTEGER(42)`, `TOKEN_SEMICOLON`).  
   Purpose: Simplify parsing by abstracting away whitespace and raw characters.  


2. **Syntactic Analysis (Parser)**  
   Validates the sequence of tokens against C grammar rules (e.g., "return <integer>;").  
   Output: An Abstract Syntax Tree (AST) representing the structure of the code.  
   Reference: §1-233 emphasizes recursive descent parsing for simplicity in early stages.  


3. **Intermediate Representation (IR) Generation**  
   Converts the AST into a platform-agnostic IR (TACKY in this case: `Return 42`).  
   Purpose: Enables optimizations and simplifies code generation for different architectures.  


4. **Code Generation**  
   Converts IR to target architecture (x64 assembly here).  
   Key Notes:  
   - Follows x64 calling conventions: return values are stored in `%eax` (§1-233).  
   - **Platform Differences**:  
     - Linux uses `.text` for the executable section and `main` as the entry point.  
     - macOS requires `.section __TEXT,__text` for the executable section and prefixes C functions with `_` (e.g., `_main` instead of `main`) due to name mangling.  
   These differences ensure the linker recognizes the entry point (`main`/`_main`) on each OS.