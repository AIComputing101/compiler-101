# Module 1: Minimal C Compiler – Build the Foundation of Compiler Design  


## 1.1 Module Objective  
The goal of this module is to construct a **minimal but functional C compiler** that handles the simplest C programs—specifically, programs with a single `main` function that returns integer constants or nested unary expressions (e.g., `return ~(-2);`). This compiler will implement the **four core compiler passes** defined in Chapter 1 of *Writing a C Compiler*:  
1. Lexer (tokenization): Convert raw C source code into a list of syntactic "tokens."  
2. Parser (AST construction): Transform tokens into a hierarchical Abstract Syntax Tree (AST) that captures program logic.  
3. Intermediate Representation (IR) generation: Translate the AST into TACKY (Three-Address Code for C-like languages), a low-level, uniform format that decouples frontend and backend.  
4. Assembly generation & code emission: Convert TACKY to x64 assembly (AT&T syntax) and write the final executable assembly file.  

By the end of this module, you will understand the universal "frontend → IR → backend" compiler architecture (Chapter 1) and how to map high-level C syntax to hardware-specific instructions—skills that scale to all advanced compilers (e.g., C++, GPU, quantum).


## 1.2 Core Background: The Four Compiler Passes  
Chapter 1 of *Writing a C Compiler* emphasizes that even minimal compilers rely on four sequential passes to avoid monolithic, hard-to-maintain code. Each pass has a clear responsibility, and outputs from one pass serve as inputs to the next. This modularity is critical: as you extend the compiler to support more features (e.g., loops, functions) in later modules, you will only need to modify or extend individual passes, not rewrite the entire system.  

Figure 1-1 (adapted from Chapter 1) illustrates the workflow:  
```
C source code (program.c)  
→ [Lexer] → Token list  
→ [Parser] → AST  
→ [IR Generation] → TACKY  
→ [Assembly Generation + Code Emission] → x64 Assembly (program.s)  
```  

Figure: Core Workflow: Source Code → Tokenization → AST → TACKY IR → Assembly
```
[C Source Code]  ← Input (Oval)
       ↓ "Raw C code (e.g., `int main(){return 2+3;}`)"
┌─────────────────────────────┐
│ Frontend: Lexer             │  ← Light Blue (Frontend)
│ (Tokenization)              │
└────────────────┬────────────┘
                 ↓ "C Tokens (e.g., [IntKeyword, Identifier("main"), ...])"
┌─────────────────────────────┐
│ Frontend: Parser            │  ← Light Blue (Frontend)
│ (AST Construction)          │
└────────────────┬────────────┘
                 ↓ "Abstract Syntax Tree (AST) (e.g., ReturnNode → BinaryNode(Add, 2, 3))"
┌─────────────────────────────┐
│ IR Generation: TACKY IR     │  ← Light Green (IR)
│ (3-Address Code)            │
└────────────────┬────────────┘
                 ↓ "TACKY IR (e.g., `tmp0=2+3; return tmp0;`)"
┌─────────────────────────────┐
│ Backend: Assembly Generator │  ← Light Orange (Backend)
│ (x64 Assembly)              │
└────────────────┬────────────┘
                 ↓ "x64 Assembly"
[x64 Assembly File (.s)]  ← Output (Oval)
```

## 1.3 Pass 1: Lexer – Tokenize Source Code  
The lexer (or "tokenizer") is the compiler’s "reading comprehension" stage: it scans the input C file character by character, skips whitespace, and groups characters into **tokens**—the smallest syntactic units of C. Tokens include keywords (`int`, `return`), identifiers (`main`), constants (`2`), punctuation (`;`, `{`), and operators (`-`, `~`).  

### 1.3.1 Defining Tokens (Chapter 1, Table 1-1)  
Chapter 1 provides a precise list of tokens for the minimal C subset, along with their regular expressions (Perl Compatible Regex, PCRE) to enable pattern matching. For example:  

| Token Type       | Description                  | Regular Expression | Example       |  
|------------------|------------------------------|--------------------|---------------|  
| Identifier       | Variable/function names      | `[a-zA-Z\_]\w*\b`   | `main`, `x`   |  
| Integer Constant | Numeric values               | `[0-9]+\b`          | `2`, `100`    |  
| `int` Keyword    | Type specifier               | `int\b`            | `int`         |  
| `return` Keyword | Return statement marker      | `return\b`         | `return`      |  
| Punctuation      | Delimiters                   | `\(`, `\)`, `{`, `}`, `;` | `(`, `;` |  
| Unary Operators  | Negation (`-`), bitwise complement (`~`) | `-`, `~` | `-`, `~` |  

Critical rule from Chapter 1: **Longest Match First**. For example, the input `--` must be parsed as a single `decrement` token (not two `negation` tokens), and `<=` as a single `less-or-equal` token (not `<` + `=`). This ensures the lexer does not misinterpret C syntax.  

### 1.3.2 Lexer Implementation (Chapter 1, Listing 1-3)  
The lexer’s logic is straightforward, as outlined in the pseudocode from Chapter 1:  

```pseudocode
while input is not empty:
    if input starts with whitespace:
        trim whitespace from start of input
    else:
        // Find the longest token matching the start of input
        longest_match = ""
        matched_token = None
        for regex in token_regex_list:
            match = find_regex_match_at_start(input, regex)
            if match.length > longest_match.length:
                longest_match = match
                matched_token = corresponding_token_type
        if no matched_token:
            raise Error("Invalid token: " + input[:10])  // Truncate for readability
        // Convert match to token and add to list
        token_list.append(create_token(matched_token, longest_match.value))
        // Remove matched substring from input
        input = input[longest_match.length:]
```  

Key implementation details from Chapter 1:  
- **Word Boundaries (`\b`)**: Ensure identifiers/keywords do not bleed into adjacent characters. For example, `main123` is a valid identifier, but `123main` is not (the lexer will throw an error, as `123` is a constant followed by an invalid token `main`).  
- **No Whitespace Tokens**: Whitespace (spaces, newlines) is only used to separate tokens—never included as a token itself (unlike Python, where whitespace affects scope).  

### 1.3.3 Lexer Test (Chapter 1, "Test the Lexer")  
To validate the lexer, use the book’s test suite (`writing-a-c-compiler-tests` repository) with the `--stage lex` flag. The test suite includes:  
- **Valid Lex Tests**: Programs with legal tokens (e.g., `int main(void) { return 2; }`).  
- **Invalid Lex Tests**: Programs with invalid tokens (e.g., `123main`, `@x`).  

Run the test command (Chapter 1):  
```bash
./test_compiler /path/to/your_compiler --chapter 1 --stage lex
```  

A passing test means the lexer correctly generates tokens for valid input and throws errors for invalid input.


## 1.4 Pass 2: Parser – Build an Abstract Syntax Tree (AST)  
The parser takes the lexer’s token list and constructs an **Abstract Syntax Tree (AST)**—a tree data structure that represents the *logic* of the program, not its syntax. For example, the code `return 2 + 3;` becomes an AST with a `Return` node whose child is an `Add` node (with children `Constant(2)` and `Constant(3)`). Syntax details like parentheses or whitespace are discarded, as they do not affect program behavior.  

### 1.4.1 AST Definition (Chapter 1, Listing 1-5)  
Chapter 1 uses the **Zephyr Abstract Syntax Description Language (ASDL)**—a language-agnostic notation—to define the AST structure for the minimal C subset. ASDL is ideal because it clearly specifies node types and their children. The ASDL for our minimal compiler is:  

```asdl
# Root node: entire program (one function definition)
program = Program(function_definition)

# Function definition: name + body (single statement)
function_definition = Function(identifier name, statement body)

# Statement: only return statements (for now)
statement = Return(exp)

# Expression: only integer constants or unary operations (for now)
exp = Constant(int) | Unary(unary_operator, exp)

# Unary operators: negation (-) or bitwise complement (~)
unary_operator = Complement | Negate
```  

- **`identifier`**: A built-in ASDL type for names (e.g., `main`—effectively a string, but distinguished from string literals).  
- **`int`**: A built-in type for integer values.  
- **`|`**: Separates alternative node constructors (e.g., an `exp` can be a `Constant` or a `Unary`).  

In your implementation (e.g., Python, Rust, OCaml), you will map this ASDL to language-specific data structures. For example, in Python, you might use classes:  
```python
class Program:
    def __init__(self, function_def):
        self.function_def = function_def

class Function:
    def __init__(self, name, body):
        self.name = name  # str (identifier)
        self.body = body  # Statement

class Return:  # Statement subclass
    def __init__(self, exp):
        self.exp = exp  # Exp

class Constant:  # Exp subclass
    def __init__(self, value):
        self.value = value  # int

class Unary:  # Exp subclass
    def __init__(self, op, inner_exp):
        self.op = op  # "Complement" or "Negate"
        self.inner_exp = inner_exp  # Exp
```  

### 1.4.2 Formal Grammar (Chapter 1, Listing 1-6)  
To guide parsing, Chapter 1 provides a **formal grammar** in Extended Backus-Naur Form (EBNF)—a notation that defines how tokens combine into valid C constructs. The grammar for the minimal subset is:  

```ebnf
<program>        ::= <function>
<function>       ::= "int" <identifier> "(" "void" ")" "{" <statement> "}"
<statement>      ::= "return" <exp> ";"
<exp>            ::= <int> | <unop> <exp> | "(" <exp> ")"
<unop>           ::= "-" | "~"
<identifier>     ::= ? Identifier token (e.g., main) ?
<int>            ::= ? Integer constant token (e.g., 2) ?
```  

- **Non-terminal symbols**: Wrapped in `< >` (e.g., `<program>`, `<exp>`)—represent language constructs.  
- **Terminal symbols**: Quoted (e.g., `"int"`, `"("`) or described (e.g., `<identifier>`)—represent tokens.  
- **`::=`**: Means "is defined as."  
- **`|`**: Means "or" (alternative constructs).  

For example, the rule `<exp> ::= <int> | <unop> <exp>` means an expression can be either a constant (e.g., `2`) or a unary operator applied to another expression (e.g., `-2`, `~(3)`).  

### 1.4.3 Recursive Descent Parsing (Chapter 1, Listing 1-7)  
Chapter 1 advocates **recursive descent parsing**—a handwritten parsing technique where each non-terminal symbol (e.g., `<program>`, `<exp>`) has a dedicated function. This approach is intuitive, debuggable, and aligns with the AST’s hierarchical structure.  

The core parsing functions for the minimal compiler are:  
1. `parse_program(tokens)`: Parses the entire program (root node).  
2. `parse_function(tokens)`: Parses a function definition (e.g., `int main(void) { ... }`).  
3. `parse_statement(tokens)`: Parses a return statement (e.g., `return 2;`).  
4. `parse_exp(tokens)`: Parses an expression (e.g., `~(-2)`).  

Chapter 1 provides pseudocode for `parse_statement` and a helper `expect` function (to validate expected tokens):  

```pseudocode
# Parses a return statement (e.g., "return 2;")
def parse_statement(tokens):
    # Step 1: Expect a "return" keyword
    expect("return", tokens)
    # Step 2: Parse the return expression (e.g., 2, ~(-2))
    return_exp = parse_exp(tokens)
    # Step 3: Expect a semicolon to end the statement
    expect(";", tokens)
    # Step 4: Return the Return AST node
    return Return(return_exp)

# Helper: Validates the next token matches "expected"; throws error if not
def expect(expected_token_type, tokens):
    if tokens is empty:
        raise Error("Unexpected end of input (expected " + expected_token_type + ")")
    actual_token = tokens.pop(0)  # Remove token from list
    if actual_token.type != expected_token_type:
        raise Error(f"Syntax error: Expected {expected_token_type}, got {actual_token.type}")
```  

For `parse_exp`, Chapter 2 extends the logic to handle unary expressions (e.g., `~(-2)`). The key is recursion: when parsing a unary operator (e.g., `~`), the function recursively parses the inner expression (e.g., `-2`):  

```pseudocode
def parse_exp(tokens):
    next_token = peek(tokens)  # Look at next token without removing it
    if next_token.type == "int":
        # Case 1: Expression is a constant (e.g., 2)
        int_token = tokens.pop(0)
        return Constant(int(int_token.value))
    elif next_token.type in ["negate", "complement"]:
        # Case 2: Expression is a unary operation (e.g., -2, ~3)
        op_token = tokens.pop(0)
        inner_exp = parse_exp(tokens)  # Recursively parse inner expression
        op = "Negate" if op_token.type == "negate" else "Complement"
        return Unary(op, inner_exp)
    elif next_token.type == "lparen":
        # Case 3: Expression is wrapped in parentheses (e.g., (2))
        expect("lparen", tokens)
        inner_exp = parse_exp(tokens)
        expect("rparen", tokens)
        return inner_exp
    else:
        raise Error(f"Invalid expression: Unexpected token {next_token.type}")
```  

### 1.4.4 AST Example (Chapter 1, Figure 1-2)  
For the program `int main(void) { return ~(-2); }`, the parser generates the following AST (hierarchical structure):  
```
Program(
    Function(
        name="main",
        body=Return(
            exp=Unary(
                op="Complement",
                inner_exp=Unary(
                    op="Negate",
                    inner_exp=Constant(value=2)
                )
            )
        )
    )
)
```  

This AST captures the program’s logic: "In function `main`, return the bitwise complement of the negation of 2." Syntax like `int`, `void`, and braces are discarded—they served only to guide parsing.  

### 1.4.5 Parser Test (Chapter 1, "Test the Parser")  
Validate the parser with the test suite’s `--stage parse` flag. The tests ensure:  
- Valid programs (e.g., `return ~(-2);`) produce correct ASTs.  
- Invalid programs (e.g., `return 2`—missing semicolon) throw syntax errors.  

Run the test command:  
```bash
./test_compiler /path/to/your_compiler --chapter 1 --stage parse
```  

For debugging, Chapter 1 recommends writing a **pretty-printer**—a function that prints the AST in human-readable format (e.g., the hierarchical structure above). This helps verify the parser’s output matches expectations.


## 1.5 Pass 3: IR Generation – Translate AST to TACKY  
Chapter 2 introduces **TACKY** (Three-Address Code for C-like languages)—a low-level Intermediate Representation (IR) that bridges the AST (high-level logic) and assembly (hardware-specific code). TACKY is critical for two reasons:  
1. **Decoupling**: The AST is tied to C syntax, while TACKY is a generic format that can be translated to any ISA (x64, ARM, etc.).  
2. **Simplicity**: TACKY avoids nested expressions—every instruction uses at most three operands (constants or temporary variables), making it easy to convert to assembly.  

### 1.5.1 TACKY Definition (Chapter 2, Listing 2-9)  
Chapter 2 defines TACKY using ASDL, extending the minimal subset to handle unary expressions:  

```asdl
# Root node: entire program (one function)
program = Program(function_definition)

# Function: name + list of TACKY instructions
function_definition = Function(identifier name, instruction* body)

# TACKY Instructions: return, unary operations, copies, jumps (for later)
instruction = Return(val) 
            | Unary(unary_operator, val src, val dst)  # src → op → dst
            | Copy(val src, val dst)                  # src → dst
            | Jump(identifier target)                 # Unconditional jump (for later)
            | Label(identifier)                      # Mark jump target (for later)

# Values: constants or temporary variables
val = Constant(int) | Var(identifier)  # Var = temporary (e.g., tmp.0)

# Unary operators (same as AST)
unary_operator = Complement | Negate
```  

Key TACKY rule: **Every operation’s result is stored in a temporary variable** (e.g., `tmp.0`, `tmp.1`). For example, the AST `Unary(Complement, Unary(Negate, Constant(2)))` (i.e., `~(-2)`) becomes two TACKY instructions:  
1. `Unary(Negate, Constant(2), Var("tmp.0"))` → Compute `-2`, store in `tmp.0`.  
2. `Unary(Complement, Var("tmp.0"), Var("tmp.1"))` → Compute `~tmp.0`, store in `tmp.1`.  
3. `Return(Var("tmp.1"))` → Return the final result.  

### 1.5.2 TACKY Generation Logic (Chapter 2, Listing 2-10)  
Chapter 2 provides pseudocode for `emit_tacky`—a recursive function that traverses the AST and appends TACKY instructions to a list. The function returns a `val` (Constant or Var) representing the result of the current AST node.  

For expressions (the core of TACKY generation):  
```pseudocode
# Traverses an AST exp node, emits TACKY instructions, returns TACKY val
def emit_tacky(exp, instructions):
    match exp with:
        | Constant(c):
            # Case 1: AST Constant → TACKY Constant (no instructions needed)
            return Constant(c)
        | Unary(op, inner_exp):
            # Case 2: AST Unary → TACKY Unary (needs temporary variable)
            # Step 1: Emit TACKY for inner expression (e.g., -2)
            src_val = emit_tacky(inner_exp, instructions)
            # Step 2: Generate unique name for temporary variable (e.g., tmp.0)
            dst_name = make_temporary()  # Uses global counter: tmp.0, tmp.1, ...
            dst_val = Var(dst_name)
            # Step 3: Convert AST operator to TACKY operator
            tacky_op = "Negate" if op == "Negate" else "Complement"
            # Step 4: Append TACKY Unary instruction
            instructions.append(Unary(tacky_op, src_val, dst_val))
            # Step 5: Return the temporary variable (holds result)
            return dst_val
```  

For statements (e.g., `Return`):  
```pseudocode
# Traverses an AST statement node, emits TACKY instructions
def emit_tacky_statement(statement, instructions):
    match statement with:
        | Return(exp):
            # Step 1: Emit TACKY for the return expression
            result_val = emit_tacky(exp, instructions)
            # Step 2: Append TACKY Return instruction
            instructions.append(Return(result_val))
```  

For functions (the top-level):  
```pseudocode
# Traverses an AST function node, emits TACKY function definition
def emit_tacky_function(function, instructions):
    # Step 1: Emit TACKY for the function body (return statement)
    emit_tacky_statement(function.body, instructions)
    # Step 2: Return TACKY Function node
    return Function(function.name, instructions)
```  

### 1.5.3 TACKY Example (Chapter 2)  
For the AST of `return ~(-2);`, `emit_tacky` generates the following TACKY instructions:  
```tacky
# Step 1: Compute -2 → tmp.0
Unary(Negate, Constant(2), Var("tmp.0"))
# Step 2: Compute ~tmp.0 → tmp.1
Unary(Complement, Var("tmp.0"), Var("tmp.1"))
# Step 3: Return tmp.1
Return(Var("tmp.1"))
```  

### 1.5.4 TACKY Test (Chapter 2, "Test the TACKY Generation Stage")  
Validate TACKY generation with the test suite’s `--stage tacky` flag. The tests ensure the compiler emits valid TACKY for all valid input programs (no syntax/semantic errors).  

Run the test command:  
```bash
./test_compiler /path/to/your_compiler --chapter 2 --stage tacky
```  


## 1.6 Pass 4: Assembly Generation & Code Emission  
The final pass converts TACKY to **x64 assembly** (AT&T syntax) and writes the assembly to a `.s` file. Chapter 2 breaks this into three sub-passes to handle complexity:  
1. **TACKY → Assembly AST**: Convert TACKY instructions to an assembly-specific AST (e.g., TACKY `Unary(Negate, ...)` → assembly `Neg` instruction).  
2. **Pseudoregister Replacement**: Replace TACKY temporary variables (`Var("tmp.0")`) with stack addresses (e.g., `-4(%rbp)`).  
3. **Instruction Fix-Up**: Rewrite invalid assembly instructions (e.g., `movl -4(%rbp), -8(%rbp)` → use a scratch register `%r10d`).  
4. **Code Emission**: Print the assembly AST to a file in AT&T syntax.  

### 1.6.1 Step 1: TACKY → Assembly AST (Chapter 2, Listing 2-11)  
Chapter 2 defines an **assembly AST** to represent x64 instructions and operands. The ASDL for the minimal compiler is:  

```asdl
# Root node: entire assembly program
program = Program(function_definition)

# Function: name + list of assembly instructions
function_definition = Function(identifier name, instruction* instructions)

# Assembly Instructions: mov, unary (neg/not), allocate stack, return
instruction = Mov(operand src, operand dst)    # src → dst
            | Unary(unary_operator, operand)  # operand → op → operand
            | AllocateStack(int)              # subq $int, %rsp (allocate stack space)
            | Ret                             # Return to caller

# Operands: immediate values, registers, pseudoregisters, stack addresses
operand = Imm(int)          # Immediate (e.g., $2)
        | Reg(reg)          # Hardware register (e.g., %eax)
        | Pseudo(identifier)# Temporary variable (e.g., tmp.0)
        | Stack(int)        # Stack address (e.g., -4(%rbp))

# Registers: Only %eax (return value) and %r10d (scratch) for now
reg = AX | R10

# Unary operators: neg (negate), not (bitwise complement)
unary_operator = Neg | Not
```  

Chapter 2 provides a table (Table 2-3) to map TACKY instructions to assembly AST nodes. For example:  
- TACKY `Unary(Negate, src, dst)` → Assembly `Mov(src, dst)` + `Unary(Neg, dst)`.  
- TACKY `Return(val)` → Assembly `Mov(val, Reg(AX))` + `Ret`.  

For the TACKY instruction `Unary(Negate, Constant(2), Var("tmp.0"))`, the assembly AST is:  
```
[
    Mov(Imm(2), Pseudo("tmp.0")),  # Move 2 to tmp.0
    Unary(Neg, Pseudo("tmp.0"))    # Negate tmp.0 → -2
]
```  

### 1.6.2 Step 2: Pseudoregister Replacement (Chapter 2)  
TACKY temporary variables (`Pseudo("tmp.0")`) are not valid in assembly—they must be mapped to **stack addresses** (relative to the base pointer `%rbp`). Chapter 2 explains that x64 functions use a **stack frame** to store local variables and temporaries. The stack grows toward lower memory addresses, so temporary variables are assigned negative offsets from `%rbp` (e.g., `tmp.0` → `-4(%rbp)`, `tmp.1` → `-8(%rbp)`).  

The pseudocode for this sub-pass is:  
```pseudocode
def replace_pseudoregisters(assembly_ast):
    # Map: TACKY var name → stack offset (e.g., "tmp.0" → -4)
    var_to_offset = {}
    next_offset = -4  # Start at -4 (4 bytes per int)
    # Traverse assembly AST to replace Pseudo operands
    for instruction in assembly_ast.instructions:
        for operand in instruction.operands:
            if operand.type == "Pseudo":
                var_name = operand.value
                if var_name not in var_to_offset:
                    var_to_offset[var_name] = next_offset
                    next_offset -= 4  # Next variable: -8, -12, etc.
                # Replace Pseudo with Stack(offset)
                operand = Stack(var_to_offset[var_name])
    # Add AllocateStack instruction to reserve stack space
    stack_size = abs(next_offset)  # e.g., next_offset = -8 → stack_size = 8
    assembly_ast.instructions.insert(0, AllocateStack(stack_size))
    return assembly_ast
```  

For the TACKY temporary `tmp.0`, this sub-pass replaces `Pseudo("tmp.0")` with `Stack(-4)` (i.e., `-4(%rbp)` in assembly).  

### 1.6.3 Step 3: Instruction Fix-Up (Chapter 2)  
Chapter 2 notes that many x64 instructions cannot use **memory addresses for both operands**. For example, `movl -4(%rbp), -8(%rbp)` (copy from one stack address to another) is invalid—at least one operand must be a register or immediate. To fix this, we use a **scratch register** (`%r10d`, a 32-bit register not used for return values or arguments):  

```pseudocode
def fix_instructions(assembly_ast):
    fixed_instructions = []
    for instruction in assembly_ast.instructions:
        if instruction.type == "Mov" and 
           instruction.src.type == "Stack" and 
           instruction.dst.type == "Stack":
            # Fix: Mov Stack → Reg → Stack
            fixed_instructions.append(Mov(instruction.src, Reg(R10)))
            fixed_instructions.append(Mov(Reg(R10), instruction.dst))
        else:
            # No fix needed
            fixed_instructions.append(instruction)
    assembly_ast.instructions = fixed_instructions
    return assembly_ast
```  

For the invalid instruction `movl -4(%rbp), -8(%rbp)`, this sub-pass generates two valid instructions:  
```asm
movl -4(%rbp), %r10d  # Stack → Reg
movl %r10d, -8(%rbp)  # Reg → Stack
```  

### 1.6.4 Step 4: Code Emission (Chapter 2, Tables 2-6 to 2-9)  
The final step is to print the assembly AST to a file in **AT&T syntax**—the standard for x64 assembly on Linux/macOS. Chapter 2 provides tables defining how to format each assembly construct. Key rules from the tables:  
- **Registers**: Prefix with `%` (e.g., `Reg(AX)` → `%eax`, `Reg(R10)` → `%r10d`).  
- **Immediates**: Prefix with `$` (e.g., `Imm(2)` → `$2`).  
- **Stack Addresses**: Format as `offset(%rbp)` (e.g., `Stack(-4)` → `-4(%rbp)`).  
- **Instructions**: Use suffixes to indicate operand size (e.g., `movl` for 32-bit integers, `subq` for 64-bit stack operations).  
- **Function Prologue/Epilogue**: Every function needs a prologue (set up stack frame) and epilogue (tear down stack frame):  
  - Prologue: `pushq %rbp; movq %rsp, %rbp` (save caller’s base pointer, set current base pointer).  
  - Epilogue: `movq %rbp, %rsp; popq %rbp` (restore stack pointer, restore caller’s base pointer) before `ret`.  

#### Example: Final Assembly for `return ~(-2);` (Chapter 2, Listing 2-2)  
After all sub-passes, the assembly AST for `return ~(-2);` is emitted as:  

```asm
# Declare main as global (for linker to find)
.globl main
main:
    # Function Prologue: Set up stack frame
    pushq %rbp
    movq %rsp, %rbp
    # Allocate 8 bytes of stack space (for tmp.0 and tmp.1)
    subq $8, %rsp
    # Step 1: tmp.0 = 2 → -4(%rbp) = 2
    movl $2, -4(%rbp)
    # Step 2: tmp.0 = -tmp.0 → negate -4(%rbp)
    negl -4(%rbp)
    # Step 3: tmp.1 = tmp.0 → copy -4(%rbp) to -8(%rbp) (uses %r10d)
    movl -4(%rbp), %r10d
    movl %r10d, -8(%rbp)
    # Step 4: tmp.1 = ~tmp.1 → complement -8(%rbp)
    notl -8(%rbp)
    # Step 5: Return tmp.1 → move -8(%rbp) to %eax (return register)
    movl -8(%rbp), %eax
    # Function Epilogue: Tear down stack frame
    movq %rbp, %rsp
    popq %rbp
    # Return to caller (crt0 runtime)
    ret
# Linux-only: Mark stack as non-executable (security)
.section .note.GNU-stack,"",@progbits
```  

### 1.6.5 Assembly Test (Chapter 2, "Test the Whole Compiler")  
To validate the final compiler, use the test suite’s `--chapter 2` flag. The tests:  
1. Compile the C program to assembly.  
2. Assemble and link the assembly to an executable (using `gcc`).  
3. Run the executable and check the exit code (e.g., `return ~(-2);` exits with code `1`, since `~(-2) = 1` in two’s complement).  

Run the test command:  
```bash
./test_compiler /path/to/your_compiler --chapter 2
```  

To verify manually, compile and run the assembly file:  
```bash
# Assemble and link: gcc assembly.s -o program
# Run: ./program
# Check exit code: echo $? → Should print 1
```  


## 1.7 Summary: Why the Minimal Compiler Matters  
Chapter 1 and 2 of *Writing a C Compiler* emphasize that the minimal compiler is not just a "toy"—it is the **foundation of all advanced compilers**. Every feature added in later modules (e.g., loops, functions, C++, GPU support) builds on these four core passes:  
- Lexers scale to handle C++ keywords (e.g., `class`, `template`) or GPU-specific syntax (e.g., CUDA’s `__global__`).  
- Parsers extend to handle control flow (e.g., `if`, `while`) or quantum circuits (e.g., CUDA-Q’s `__qpu__`).  
- IRs (like TACKY) evolve to support parallelism (e.g., Triton’s custom IR for GPU threads) or quantum operations (e.g., OpenQASM).  
- Assembly backends adapt to new ISAs (e.g., AMD GCN, quantum processor instructions).  

By mastering this minimal compiler, you gain a "mental model" of how compilers translate high-level code to machine code—a skill that applies to every compiler you will encounter or build.