# Module 2: Standard C Compiler – Extending to Real-World C Features  
*(Exclusively Referenced from* Writing a C Compiler (2024.7) by Nora Sandler *)*  


## 2.1 Module Objective  
The minimal compiler from Module 1 handles only the simplest C programs: a single `main` function returning integer constants or unary expressions (e.g., `return ~(-2);`). To qualify as a "standard" C compiler, we must extend it to support **real-world C features** outlined in Chapters 5–10 of *Writing a C Compiler*:  
- Variables (declaration, initialization, assignment).  
- Control flow (`if` statements, `while` loops, `break`/`continue`).  
- Functions (parameters, return values, nested calls).  
- File scope and storage-class specifiers (`static`, `extern`).  

This module focuses on adding these features by enhancing the four core compiler passes (lexer, parser, IR generator, assembly backend) while maintaining the modular design from Module 1. By the end, the compiler will handle programs like:  

```c
#include <stdio.h>  // For illustration (we’ll add basic stdio later)

int global_var = 42;  // File-scope variable

static int helper(int x) {  // Static function (internal linkage)
    if (x < 0) return -x;   // if statement
    else return x;
}

int main(void) {
    int a = 5;              // Local variable
    int b;                  // Uninitialized local variable
    b = helper(a - 10);     // Function call with expression
    while (b > 0) {         // while loop
        b--;
        if (b == 2) break;  // break statement
    }
    return b;
}
```  


## 2.2 New Stage: Semantic Analysis – Ensuring Program Logic is Valid  
The minimal compiler (Module 1) only checks for syntax errors (e.g., missing semicolons). A standard C compiler requires **semantic analysis** (Chapter 5) to validate the *logic* of the program, such as:  
- Variables are declared before use.  
- Variables are not redeclared in the same scope.  
- Operators are applied to compatible types (e.g., no adding a `int` to a `float`).  
- Functions are called with the correct number/type of arguments.  

### 2.2.1 The Symbol Table (Chapter 5, Listing 5-1)  
Semantic analysis relies on a **symbol table**—a data structure that tracks identifiers (variables, functions) and their properties (type, scope, linkage). Chapter 5 defines the symbol table as a stack of "scopes," where each scope corresponds to a block (e.g., function body, `if` block). When looking up an identifier, the compiler checks the innermost scope first, then outer scopes.  

The symbol table entries (per Chapter 5) include:  
- `name`: The identifier (e.g., `a`, `helper`).  
- `type`: Data type (e.g., `int`).  
- `scope`: The block where the identifier is declared (e.g., `main` function, global).  
- `storage_class`: For variables: `auto` (default), `static`, `extern`. For functions: `static`, `extern` (default).  
- `is_function`: Boolean indicating if the entry is a function (vs. variable).  
- `parameters`: For functions: list of parameter types (e.g., `(int)` for `helper`).  

### 2.2.2 Scope Resolution (Chapter 5, "Scope Rules")  
C uses **lexical scoping**: an identifier is visible from its declaration to the end of its enclosing block. Chapter 5 provides rules for resolving identifiers:  
1. **Block Scope**: Variables declared inside `{ }` (e.g., function bodies, `if` blocks) are visible only within that block.  
2. **File Scope**: Variables/functions declared outside all blocks (global) are visible from their declaration to the end of the file.  
3. **No Redeclaration**: An identifier cannot be declared twice in the same scope.  

Example (from Chapter 5):  
```c
int x = 1;  // File scope
int main(void) {
    int x = 2;  // Block scope (hides global x)
    {
        int x = 3;  // Inner block scope (hides main’s x)
        return x;  // Returns 3
    }
}
```  

The semantic analyzer uses the symbol table to enforce these rules. If a variable is used before declaration (e.g., `return y;` where `y` is never declared), it throws a "use of undeclared identifier" error.  

### 2.2.3 Type Checking (Chapter 5, "Type Checking")  
Chapter 5 emphasizes that C is a **statically typed** language: types are checked at compile time, not runtime. The semantic analyzer validates that operations are applied to compatible types:  
- Arithmetic operators (`+`, `-`, `*`, `/`) require integer or floating-point operands (no mixing `int` and pointers).  
- Assignment requires the left-hand side (lvalue) to be a modifiable variable (e.g., `42 = x;` is invalid).  
- Function calls require arguments to match the parameter types (e.g., calling `helper("hello")` when `helper` expects `int` is invalid).  

Example type error (from Chapter 5):  
```c
int main(void) {
    int a = 5;
    return a + "hello";  // Error: cannot add int and pointer-to-char
}
```  


## 2.3 Variables: Declaration, Initialization, and Assignment  
The minimal compiler (Module 1) only handles constants. A standard compiler must support variables: named storage locations with a type and value.  

### 2.3.1 Parsing Variable Declarations (Chapter 5, Listing 5-3)  
To parse variable declarations (e.g., `int a = 5;`, `int b;`), we extend the parser with new grammar rules (Chapter 5):  

```ebnf
<statement>      ::= "return" <exp> ";" 
                    | <declaration>  // New: variable declaration
<declaration>    ::= "int" <identifier> ( "=" <exp> )? ";"  // ( "=" <exp> )? = optional initialization
```  

The parser’s `parse_statement` function is updated to handle declarations:  

```pseudocode
def parse_statement(tokens, symbol_table):
    next_token = peek(tokens)
    if next_token.type == "return":
        # Existing logic for return statements
        ...
    elif next_token.type == "int":
        # Parse declaration: "int id = exp;" or "int id;"
        expect("int", tokens)
        id_token = expect("identifier", tokens)
        id_name = id_token.value
        # Check for redeclaration in current scope
        if symbol_table.lookup(id_name, current_scope):
            raise Error(f"Redeclaration of {id_name}")
        # Parse optional initialization
        initializer = None
        if peek(tokens).type == "assign":
            expect("assign", tokens)  # "="
            initializer = parse_exp(tokens)
        expect(";", tokens)
        # Add to symbol table
        symbol_table.add(
            name=id_name,
            type="int",
            scope=current_scope,
            storage_class="auto",  # Default for local variables
            is_function=False
        )
        # Return Declaration AST node
        return Declaration(name=id_name, initializer=initializer)
    else:
        raise Error(f"Unexpected token: {next_token.type}")
```  

### 2.3.2 Assignment Expressions (Chapter 5, "Assignment Operators")  
C allows variables to be assigned values after declaration (e.g., `b = a + 3;`). We extend the expression grammar (Chapter 5) to include assignments:  

```ebnf
<exp>            ::= <unop> <exp> 
                    | <exp> <assign_op> <exp>  // New: assignment (e.g., =, +=)
                    | <int> | "(" <exp> ")"
<assign_op>      ::= "=" | "+=" | "-=" | "*=" | "/="  // Simplified for now
```  

The parser’s `parse_exp` function is updated to handle assignments, ensuring the left-hand side is a valid lvalue (variable):  

```pseudocode
def parse_exp(tokens, symbol_table):
    # Parse primary expression (e.g., constant, variable, unary)
    lhs = parse_primary_exp(tokens, symbol_table)
    # Check for assignment operator
    if peek(tokens).type in ["assign", "add_assign", ...]:
        op = tokens.pop(0).type
        rhs = parse_exp(tokens, symbol_table)
        # Validate lhs is a variable (lvalue)
        if not isinstance(lhs, VarNode):
            raise Error("Left-hand side of assignment must be a variable")
        # Check variable exists in symbol table
        if not symbol_table.lookup(lhs.name):
            raise Error(f"Undeclared variable: {lhs.name}")
        return AssignNode(op, lhs, rhs)
    return lhs
```  

### 2.3.3 TACKY for Variables and Assignments (Chapter 5, Listing 5-5)  
The IR generator is extended to handle `Declaration` and `Assign` AST nodes. For declarations with initializers (e.g., `int a = 5;`), TACKY generates a `Copy` instruction to initialize the variable. For assignments (e.g., `b = a + 3;`), TACKY generates instructions to compute the right-hand side, then a `Copy` to the left-hand side.  

Example TACKY for `int a = 5; int b; b = a + 3;`:  
```tacky
# Declaration with initializer: int a = 5
Copy(Constant(5), Var("a"))  # a = 5

# Assignment: b = a + 3
Unary(Add, Var("a"), Constant(3), Var("tmp.0"))  # tmp.0 = a + 3
Copy(Var("tmp.0"), Var("b"))  # b = tmp.0
```  


## 2.4 Control Flow: `if` Statements and Loops  
Control flow (conditional execution, repetition) is critical for real-world programs. Chapter 6 covers `if` statements, and Chapters 7–8 cover `while` loops, `break`, and `continue`.  

### 2.4.1 `if` Statements (Chapter 6, Listing 6-1)  
An `if` statement executes a block conditionally (e.g., `if (x > 0) { ... } else { ... }`). The grammar is extended as:  

```ebnf
<statement>      ::= ... 
                    | "if" "(" <exp> ")" <statement> ( "else" <statement> )?  // New: if/else
```  

The parser generates an `IfNode` AST, which the IR generator converts to TACKY with conditional jumps (`JumpIfZero`) and labels:  

```tacky
# TACKY for: if (a > 0) return 1; else return 0;
# Step 1: Compute a > 0 → tmp.0 (1 if true, 0 if false)
Unary(Greater, Var("a"), Constant(0), Var("tmp.0"))
# Step 2: Jump to else_label if tmp.0 is 0 (a ≤ 0)
JumpIfZero(Var("tmp.0"), Label("else_label"))
# Step 3: Then block: return 1
Return(Constant(1))
# Step 4: Jump past else block (to avoid falling through)
Jump(Label("end_label"))
# Step 5: Else block: return 0
Label("else_label")
Return(Constant(0))
# Step 6: End of if statement
Label("end_label")
```  

### 2.4.2 `while` Loops (Chapter 7, Listing 7-2)  
A `while` loop repeats a block while a condition is true (e.g., `while (b > 0) { b--; }`). The grammar is:  

```ebnf
<statement>      ::= ... 
                    | "while" "(" <exp> ")" <statement>  // New: while loop
```  

The AST `WhileNode` is converted to TACKY with a loop start label, condition check, and back jumps:  

```tacky
# TACKY for: while (b > 0) { b--; }
Label("loop_start")
# Step 1: Check condition (b > 0)
Unary(Greater, Var("b"), Constant(0), Var("tmp.0"))
# Step 2: Exit loop if condition is false
JumpIfZero(Var("tmp.0"), Label("loop_end"))
# Step 3: Loop body: b--
Unary(Subtract, Var("b"), Constant(1), Var("tmp.1"))
Copy(Var("tmp.1"), Var("b"))
# Step 4: Jump back to condition check
Jump(Label("loop_start"))
Label("loop_end")
```  

### 2.4.3 `break` and `continue` (Chapter 8, Listing 8-1)  
`break` exits a loop early; `continue` skips the rest of the current iteration. These require tracking loop labels (start and end) in the symbol table during parsing.  

- `break` generates a `Jump` to the loop’s end label.  
- `continue` generates a `Jump` to the loop’s start label (after the condition check).  

Example TACKY with `break`:  
```tacky
# TACKY for: while (b > 0) { if (b == 2) break; b--; }
Label("loop_start")
Unary(Greater, Var("b"), Constant(0), Var("tmp.0"))
JumpIfZero(Var("tmp.0"), Label("loop_end"))

# Body: if (b == 2) break
Unary(Equal, Var("b"), Constant(2), Var("tmp.1"))
JumpIfZero(Var("tmp.1"), Label("skip_break"))
Jump(Label("loop_end"))  # break
Label("skip_break")

# Rest of body: b--
Unary(Subtract, Var("b"), Constant(1), Var("tmp.2"))
Copy(Var("tmp.2"), Var("b"))

Jump(Label("loop_start"))
Label("loop_end")
```  


## 2.5 Functions: Parameters, Calls, and Return Values  
Functions enable code reuse. Chapter 9 covers function definitions, parameters, calls, and the x64 System V ABI (Application Binary Interface) for interoperability.  

### 2.5.1 Function Definitions (Chapter 9, Listing 9-1)  
A function definition has a return type, name, parameters, and body (e.g., `int helper(int x) { ... }`). The grammar is extended:  

```ebnf
<program>        ::= <function>+  // Multiple functions (not just main)
<function>       ::= "int" <identifier> "(" <parameter_list> ")" "{" <statement>* "}"
<parameter_list> ::= "void" | ( "int" <identifier> ( "," "int" <identifier> )* )?  // e.g., (int x, int y)
```  

The parser generates a `FunctionDefNode` AST, which includes parameters (added to the function’s scope in the symbol table).  

### 2.5.2 Function Calls (Chapter 9, Listing 9-3)  
A function call (e.g., `helper(a - 10)`) requires parsing arguments, checking they match parameters, and generating TACKY `Call` instructions.  

The expression grammar is extended to include calls:  

```ebnf
<exp>            ::= ... 
                    | <identifier> "(" <arg_list> ")"  // New: function call
<arg_list>       ::= <exp> ( "," <exp> )* |  // e.g., (a, b + 3)
```  

The TACKY `Call` instruction includes the function name, arguments, and a temporary to store the return value:  

```tacky
# TACKY for: b = helper(a - 10)
# Step 1: Compute argument: a - 10 → tmp.0
Unary(Subtract, Var("a"), Constant(10), Var("tmp.0"))
# Step 2: Call helper with tmp.0, store result in tmp.1
Call(Function("helper"), [Var("tmp.0")], Var("tmp.1"))
# Step 3: Assign result to b
Copy(Var("tmp.1"), Var("b"))
```  

### 2.5.3 x64 System V ABI for Function Calls (Chapter 9, Table 9-1)  
To ensure functions compiled by our compiler work with other code (e.g., the C standard library), we follow the **x64 System V ABI** (Chapter 9). Key rules:  
- **Parameter Passing**: The first 6 integer arguments are passed in registers `%rdi`, `%rsi`, `%rdx`, `%rcx`, `%r8`, `%r9` (32-bit: `%edi`, `%esi`, etc.). Extra arguments are pushed to the stack (reverse order).  
- **Return Values**: 32-bit integers are returned in `%eax`; 64-bit in `%rax`.  
- **Stack Frame**: Functions must preserve certain registers (`%rbp`, `%rbx`, `%r12`–`%r15`); others are scratch.  

Example assembly for calling `helper(a - 10)`:  
```asm
# Compute argument: a - 10 → %edi (1st argument register)
movl -4(%rbp), %edi    ; a is at -4(%rbp)
subl $10, %edi         ; %edi = a - 10

# Call helper
call helper            ; Return value in %eax

# Assign result to b (b is at -8(%rbp))
movl %eax, -8(%rbp)
```  


## 2.6 File Scope and Storage-Class Specifiers  
Chapter 10 covers variables and functions declared at file scope (outside all blocks) and storage-class specifiers that control linkage and storage duration.  

### 2.6.1 `static` – Internal Linkage (Chapter 10, "Static Storage Class")  
- **Variables**: `static int x = 0;` (file scope) is stored in the **data section** (persistent storage, initialized once) and visible only in the current file.  
- **Functions**: `static int helper(...)` is visible only in the current file (cannot be called from other files).  

Example assembly for a static file-scope variable:  
```asm
.data                   ; Data section (persistent storage)
.align 4                ; Align to 4 bytes (32-bit int)
static_x: .long 0       ; static int x = 0;
```  

### 2.6.2 `extern` – External Linkage (Chapter 10, "Extern Storage Class")  
`extern int global_var;` declares a variable defined in another file (resolved by the linker at runtime). No storage is allocated in the current file—only a symbol table entry.  

Example: If `global_var` is defined in `other.c`, our compiler emits a reference to it in assembly:  
```asm
.extern global_var      ; Declare global_var is defined elsewhere
...
movl global_var(%rip), %eax  ; Access global_var
```  


## 2.7 Testing the Standard Compiler  
Chapter 10 recommends validating the compiler with tests covering:  
- Variable scoping (e.g., local vs. global variables).  
- Control flow (e.g., `if` with true/false conditions, loops with `break`).  
- Function calls (e.g., parameters passed correctly, return values used).  
- Storage classes (e.g., `static` variables retain values between calls).  

Run the test suite with:  
```bash
./test_compiler /path/to/your_compiler --chapters 5-10
```  


## 2.8 Summary  
The standard C compiler builds on the minimal compiler’s four passes but adds critical features via:  
- **Semantic analysis** (symbol table, scope resolution, type checking) to ensure logical validity.  
- **Variables and assignments** (declarations, initialization) via extended parsing and TACKY.  
- **Control flow** (`if`, `while`, `break`) using TACKY jumps and labels.  
- **Functions** following the x64 System V ABI for interoperability.  
- **File scope** and storage classes (`static`, `extern`) to manage linkage and storage.  

These extensions transform the minimal compiler into a tool capable of handling real-world C programs, laying the groundwork for even more advanced compilers (C++, GPU, quantum) in subsequent modules.

Figure: Extension of Module 1 + Semantic Analysis + Real-World C Features
```
[C Source Code (vars/loops/funcs)]  ← Input (Oval)
       ↓ "C code with `int a=5;`, `while(a>0)`, `int add(int)`"
┌─────────────────────────────┐
│ Frontend: Lexer             │  ← Light Blue (Frontend)
│ (Adds tokens: `static`, `while`, `()`) │
└────────────────┬────────────┘
                 ↓ "Enhanced C Tokens"
┌─────────────────────────────┐
│ Frontend: Parser            │  ← Light Blue (Frontend)
│ (AST for C features:        │
│  - DeclarationNode (vars)   │
│  - WhileNode (loops)        │
│  - FuncNode (functions))    │
└────────────────┬────────────┘
                 ↓ "C-Aware AST"
┌─────────────────────────────┐
│ Semantic Analysis           │  ← Light Yellow (Optimization)
│ (New Critical Stage):       │
│  • Symbol Table (tracks var scope/type) │
│  • Scope Resolution (rejects undeclared vars) │
│  • Type Checking (validates func args) │
└────────────────┬────────────┘
                 ↓ "Validated C AST"
┌─────────────────────────────┐
│ IR Generation: TACKY IR     │  ← Light Green (IR)
│ (Adds IR for:               │
│  - `Copy(var, val)` (assignments) │
│  - `JumpIfZero` (loops)     │
│  - `Call(func, args)` (funcs)) │
└────────────────┬────────────┘
                 ↓ "C-Enhanced TACKY IR"
┌─────────────────────────────┐
│ Backend: Assembly Generator │  ← Light Orange (Backend)
│ (Follows x64 System V ABI:  │
│  - Params in %edi/%esi      │
│  - Return values in %eax)   │
└────────────────┬────────────┘
                 ↓ "ABI-Compliant x64 Assembly"
[Executable x64 Binary]  ← Output (Oval)
```