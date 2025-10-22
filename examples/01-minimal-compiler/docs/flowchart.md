# Flow Chart

Compiling Flow: return 42; → Exit Code 42

┌───────────────────┐
│ Input File        │  ← Contains the code to compile:
│ ../tests/input.txt│    "return 42;"
└───────────┬───────┘
            │
            ▼
┌───────────────────┐
│ 1. Lexer          │  ← Converts text into "tokens" (meaningful chunks)
│ (lexer.c)         │    Output: [TOKEN_RETURN, TOKEN_INTEGER(42), TOKEN_SEMICOLON, TOKEN_EOF]
└───────────┬───────┘
            │
            ▼
┌───────────────────┐
│ 2. Parser         │  ← Checks if tokens follow valid C syntax
│ (parser.c)        │    Output: AST (Abstract Syntax Tree) representing "return 42;"
└───────────┬───────┘      Structure: ReturnStmt { value = 42 }
            │
            ▼
┌───────────────────┐
│ 3. IR Generator   │  ← Converts AST to platform-agnostic "intermediate code"
│ (ir.c)            │    Output: TACKY IR node: IR_Return { value = 42 }
└───────────┬───────┘
            │
            ▼
┌───────────────────┐
│ 4. Code Generator │  ← Converts IR to machine-specific assembly code
│ (codegen.c)       │    Output: x64 Assembly (saved to temp.s)
└───────────┬───────┘      - macOS: Uses "_main" and ".section __TEXT,__text"
            │              - Linux: Uses "main" and ".text"
            ▼
┌───────────────────┐
│ 5. Assembler +    │  ← Turns assembly into an executable file
│    Linker         │    Command: gcc temp.s -o output
│ (via gcc)         │    Output: Executable file named "output"
└───────────┬───────┘
            │
            ▼
┌───────────────────┐
│ 6. Executable     │  ← When run, returns the value from "return 42;"
│ ./output          │    Output: Exit code = 42
└───────────────────┘

## Key Descriptions

Input File: The raw code to compile (e.g., return 42;).
Lexer: Breaks text into tokens (like "words" for the compiler) to simplify processing.
Parser: Ensures tokens form valid C syntax (e.g., "return" must be followed by a number and semicolon).
IR Generator: Creates intermediate code that works on any platform (not tied to x64, ARM, etc.).
Code Generator: Converts intermediate code to assembly for the target CPU/OS (handles macOS/Linux differences).
Assembler + Linker: Turns assembly into a runnable program.
Executable: When run, produces the final result (exit code 42).

This diagram is 100% text-based (no special syntax) and clearly shows how return 42; flows through each stage to become an exit code. You can copy/paste it anywhere, and it will always be readable!