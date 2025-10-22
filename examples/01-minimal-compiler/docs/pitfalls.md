# Common Pitfalls

1. **Missing `ret` Instruction**  
   - Symptom: Segmentation fault when running the executable.  
   - Cause: The code generator forgot to emit `ret` in assembly, so the CPU executes garbage after `main`.  
   - Fix: Ensure `generate_asm()` includes `ret` after setting `%eax`.  


2. **Invalid Token Order**  
   - Symptom: Parser error ("Unexpected token").  
   - Example: Input "42 return;" (wrong order) or "return 42" (missing semicolon).  
   - Fix: Validate input matches "return <integer>;".  


3. **Integer Overflow**  
   - Symptom: Incorrect return value (e.g., "return 1234567890123;" → garbage).  
   - Cause: The lexer stores integers in a 32-bit `int`.  
   - Fix: Use `long long` for larger values (extend `Token.integer_value` and AST/IR accordingly).  


4. **"Undefined Symbol _main" on macOS**  
   - Symptom: Linker error: `Undefined symbols for architecture x86_64: "_main"`.  
   - Cause: macOS requires:  
     - Executable code in the `.section __TEXT,__text` section (not just `.text`).  
     - C function names prefixed with `_` (e.g., `_main` instead of `main`).  
   - Fix: Update `codegen.c` to use these macOS-specific conventions (see walkthrough for examples).  