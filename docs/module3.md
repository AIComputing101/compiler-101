# Module 3: Modern C++ Compilers (GCC/Clang) – Extending C to Support C++’s Rich Feature Set  

Modern C++ compilers like GCC (14+) and Clang (18+) are not从头开始构建的全新 tools—they evolve from the standard C compilers outlined in Module 2, extending their core stages to handle C++’s unique features: object-oriented programming (OOP), templates, exceptions, and modern standards (C++11 to C++23). This module explores how these compilers reuse C’s foundational architecture (lexer → parser → IR → assembly) while adding layers to support C++’s complexity, drawing on principles from *Writing a C Compiler* (for C’s core stages) and *From Source Code to Machine Code* (for compiler extensibility).  


## 3.1 Module Objective  
By the end of this module, you will understand how modern C++ compilers:  
- Extend C’s lexer to recognize C++-specific tokens (e.g., `class`, `template`).  
- Expand the parser to construct ASTs for OOP, templates, and lambdas.  
- Enhance semantic analysis to validate C++ logic (e.g., virtual function overrides, template constraints).  
- Use advanced IRs (like LLVM IR) to represent C++ features (e.g., vtables, exceptions).  
- Generate assembly compliant with C++ ABIs (e.g., Itanium ABI) for interoperability.  

The focus is on **incremental extension**: C++ compilers reuse C’s infrastructure but add modules to handle C++’s richer syntax and semantics, ensuring backward compatibility with C code.  


## 3.2 Lexer: Handling C++’s Expanded Token Set  
C++ retains all C tokens but adds hundreds of new ones to support its features. The lexer (tokenizer) is extended to recognize these, building on the C lexer’s regex-based pattern matching (from Module 1).  


### 3.2.1 New Keywords and Contextual Keywords  
C++ introduces keywords for OOP, templates, and modern features. Unlike C, some C++ keywords are **contextual**—they act as keywords only in specific contexts (e.g., `override` in function declarations).  

| Category               | Examples                                                                 | Purpose                                                                 |  
|------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|  
| OOP                    | `class`, `struct`, `public`, `private`, `protected`, `virtual`, `override` | Define classes, access control, and polymorphic functions.             |  
| Templates              | `template`, `typename`, `template`, `constexpr`, `concepts` (C++20)       | Enable generic programming and compile-time computation.                |  
| Modern C++ (C++11+)    | `auto` (type deduction), `nullptr`, `lambda` (`[]`), `co_await` (C++20)   | Simplify type handling, replace `NULL`, enable anonymous functions, and async code. |  

**Example**: The keyword `override` ensures a function correctly overrides a base class virtual function. The lexer must recognize it only when used after a function declaration (e.g., `void foo() override;`), not as an identifier in other contexts.  


### 3.2.2 New Operators and Punctuators  
C++ adds operators to support its features, many with higher precedence than C’s operators. The lexer uses the "longest match" rule (from Module 1) to distinguish them:  

| Operator       | Purpose                                                                 | Example                                                                 |  
|----------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|  
| `::`           | Scope resolution (access class/namespace members).                      | `std::cout`, `MyClass::static_var`                                      |  
| `->*`          | Pointer-to-member access (call a member function via a pointer).        | `obj->*func_ptr()`                                                      |  
| `sizeof...`    | Variadic template operator (get number of template arguments).          | `template <typename... Ts> int count() { return sizeof...(Ts); }`       |  
| `=>`           | Lambda trailing return type (C++11+).                                   | `auto add = [](int a, int b) -> int { return a + b; };`                 |  

**Lexer Challenge**: Differentiating `::` from `:` (label/bitfield). The lexer checks for consecutive colons to match `::` first, ensuring `MyClass::foo` is tokenized as `Identifier("MyClass")`, `ScopeOp`, `Identifier("foo")`.  


### 3.2.3 Extended Literals  
C++ expands C’s literals to support more data types and user-defined semantics:  

- **Raw string literals**: Preserve whitespace and special characters (e.g., `R"(Line 1\nLine 2)"` avoids escaping `\n`).  
- **User-defined literals**: Extend literals with suffixes (e.g., `123_km` to represent kilometers, defined via `constexpr long double operator""_km(long double x) { return x * 1000; }`).  
- **`nullptr`**: Type-safe null pointer (replaces C’s `NULL`, which is `(void*)0`).  

The lexer uses regex to match these, e.g., `R"\([^)]*\)"` for raw strings, and `[0-9]+_([a-zA-Z_]\w*)` for user-defined numeric literals.  


## 3.3 Parser: Constructing C++-Aware ASTs  
C++ syntax is significantly more complex than C’s, requiring the parser to handle nested scopes, templates, and OOP constructs. The parser extends C’s recursive descent approach (Module 2) to build ASTs that represent these features.  


### 3.3.1 Class and Struct Declarations  
C++ classes (and structs, which are nearly identical) encapsulate data and functions. The parser generates `ClassNode` ASTs with members, access specifiers, and base classes:  

**Example Code**:  
```cpp
class Vehicle {
protected:
    int speed;
public:
    virtual void accelerate() = 0;  // Pure virtual function (abstract)
};

class Car : public Vehicle {
public:
    void accelerate() override { speed += 10; }  // Override
};
```  

**AST Structure**:  
```
TranslationUnit
├─ ClassNode(name="Vehicle", is_struct=false)
│  ├─ AccessSpecifierNode("protected")
│  │  └─ MemberVarNode(name="speed", type="int")
│  ├─ AccessSpecifierNode("public")
│  │  └─ VirtualFuncNode(
│  │       name="accelerate", 
│  │       return_type="void", 
│  │       is_pure=true  // = 0 makes it abstract
│  │     )
│  └─ BaseClasses: []
└─ ClassNode(name="Car", is_struct=false)
   ├─ BaseClasses: [BaseClassNode(name="Vehicle", access="public")]
   ├─ AccessSpecifierNode("public")
   │  └─ FuncNode(
   │       name="accelerate", 
   │       return_type="void", 
   │       is_override=true,
   │       body=...  // speed += 10
   │     )
   └─ ...
```  

The parser ensures:  
- Access specifiers (`public`/`private`/`protected`) apply to subsequent members until a new specifier is encountered.  
- Base classes are parsed with their access levels (e.g., `public Vehicle`).  


### 3.3.2 Templates: Generic Types and Functions  
Templates enable "write once, use many times" code for multiple types. The parser generates `TemplateNode` ASTs, which act as blueprints for type-specific instantiations:  

**Example Code**:  
```cpp
template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

// Instantiation: max<int>(3, 5)
```  

**AST Structure**:  
```
TemplateNode(
    parameters=[TemplateParamNode(name="T", kind="type")],
    body=FuncNode(
        name="max",
        return_type="T",
        parameters=[ParamNode(name="a", type="T"), ParamNode(name="b", type="T")],
        body=IfNode(...)  // return (a > b) ? a : b
    )
)
```  

The parser handles:  
- Template parameters (type parameters like `typename T`, non-type parameters like `int N`).  
- Variadic templates (e.g., `template <typename... Ts>` for variable numbers of types).  


### 3.3.3 Lambdas: Anonymous Functions  
Lambdas (C++11+) are inline, anonymous functions with optional captures. The parser generates `LambdaNode` ASTs to represent their logic and captured variables:  

**Example Code**:  
```cpp
int x = 5;
auto add_x = [x](int y) { return x + y; };  // Capture x by value
```  

**AST Structure**:  
```
LambdaNode(
    captures=[CaptureNode(name="x", kind="by_value")],  // [x]
    parameters=[ParamNode(name="y", type="int")],
    return_type="auto",  // Deduced as int
    body=BinaryOpNode("+", VarNode("x"), VarNode("y"))
)
```  

The parser must:  
- Parse capture lists (e.g., `[=]` captures all used variables by value, `[&]` by reference).  
- Handle trailing return types (e.g., `[](int a) -> long { return a * 2; }`).  


## 3.4 Semantic Analysis: Validating C++ Logic  
C++’s semantic analysis is far more rigorous than C’s, extending the symbol table (Module 2) to track OOP relationships, template constraints, and modern features.  


### 3.4.1 OOP-Specific Checks  
- **Virtual Function Overrides**: Ensure derived class functions correctly override base class virtual functions (same name, parameters, return type, and `const`-qualification). The `override` keyword enforces this (e.g., `void accelerate() override` must match a base `virtual void accelerate()`).  
- **Access Control**: Prevent access to `private` members from outside the class, or `protected` members from unrelated classes.  
- **Abstract Classes**: Disallow instantiation of classes with pure virtual functions (e.g., `Vehicle v;` is invalid because `Vehicle` has `= 0`).  


### 3.4.2 Template Validation  
Templates are validated in two phases (per C++’s "two-phase lookup"):  
1. **Phase 1**: Check syntax and non-dependent names (e.g., `template <typename T> void f() { int x = unknown; }` errors if `unknown` is undefined).  
2. **Phase 2**: Check dependent names during instantiation (e.g., `template <typename T> void f(T t) { t.foo(); }` errors if `T` has no `foo()` method when instantiated with `int`).  

**Example Error**:  
```cpp
template <typename T>
void add(T a, T b) {
    return a + b;  // Phase 2 error if T is a type without '+' (e.g., void*)
}

add<void*>(nullptr, nullptr);  // Compile error: '+' not defined for void*
```  


### 3.4.3 Modern C++ Checks  
- **`constexpr` Validation**: Ensure `constexpr` functions (compile-time executable) use only compile-time operations. For example:  
  ```cpp
  constexpr int square(int x) { return x * x; }  // Valid
  constexpr int bad() { int x; return x; }       // Invalid: x is uninitialized
  ```  
- **Concepts (C++20)**: Enforce template arguments meet requirements. For example:  
  ```cpp
  template <std::integral T>  // T must be an integer type (int, long, etc.)
  T double_val(T x) { return x * 2; }

  double_val(3.14);  // Error: double is not integral
  ```  


## 3.5 Intermediate Representation (IR): LLVM IR for C++  
Modern C++ compilers (Clang, GCC) use IRs like LLVM IR or GIMPLE to represent C++ features in a target-agnostic way. These IRs extend C’s TACKY (Module 1) to handle:  


### 3.5.1 Virtual Functions and Vtables  
Polymorphic calls (e.g., `Vehicle* v = new Car(); v->accelerate();`) use **vtables** (virtual tables)—arrays of function pointers stored in the data section. The IR generates vtables for classes with virtual functions:  

**LLVM IR for `Vehicle` and `Car` Vtables**:  
```llvm
@_ZTV7Vehicle = dso_local unnamed_addr constant [3 x ptr] [
    ptr null,  // Offset to top (ABI requirement)
    ptr @_ZTI7Vehicle,  // Type info (for RTTI)
    ptr null  // Pure virtual accelerate() (nullptr)
], align 8

@_ZTV3Car = dso_local unnamed_addr constant [3 x ptr] [
    ptr null,
    ptr @_ZTI3Car,  // Car's type info
    ptr @_ZN3Car10accelerateEv  // Car::accelerate()
], align 8
```  

- Each class with virtual functions has a vtable.  
- Derived classes (e.g., `Car`) override vtable entries with their own function addresses.  


### 3.5.2 Exceptions  
C++ exceptions (`try`/`catch`) are represented in IR with intrinsics for exception handling:  

**Example Code**:  
```cpp
try {
    throw std::runtime_error("Oops");
} catch (const std::exception& e) {
    // Handle error
}
```  

**LLVM IR Snippet**:  
```llvm
; Throw exception
call void @__cxa_throw(ptr %e, ptr @_ZTISt13runtime_error, ptr null)

; Catch block setup
%eh.selector = call i32 @llvm.eh.selector(...)
switch i32 %eh.selector, label %unwind [
    i32 1, label %catch  // Match std::exception
]
```  

IR intrinsics like `llvm.eh.selector` map thrown exceptions to their corresponding `catch` blocks.  


### 3.5.3 Template Instantiation  
Templates are instantiated at IR generation time, creating type-specific functions. For `max<int>(3,5)` and `max<double>(3.14, 2.7)`, the IR generates two versions of `max`:  

```llvm
; max<int>
define dso_local i32 @_Z3maxIiET_S0_S0_(i32 %a, i32 %b) {
    %cmp = icmp sgt i32 %a, %b
    %ret = select i1 %cmp, i32 %a, i32 %b
    ret i32 %ret
}

; max<double>
define dso_local double @_Z3maxIdET_S0_S0_(double %a, double %b) {
    %cmp = fcmp ogt double %a, %b
    %ret = select i1 %cmp, double %a, double %b
    ret double %ret
}
```  


## 3.6 Assembly Generation: C++ ABI Compliance  
C++ compilers generate assembly compliant with the **Itanium C++ ABI** (used by GCC/Clang) to ensure interoperability between compiled code (e.g., linking a C++ library to an executable). Key ABI rules include:  


### 3.6.1 Name Mangling  
C++ allows function overloading (multiple functions with the same name but different parameters). To avoid linker conflicts, names are **mangled** (encoded with type info):  

| Function Declaration               | Mangled Name (Itanium ABI) |  
|-------------------------------------|----------------------------|  
| `void foo(int)`                     | `_Z3fooi`                  |  
| `void foo(double)`                  | `_Z3food`                  |  
| `class Car { void accelerate(); };` | `_ZN3Car10accelerateEv`    |  

Mangling ensures `foo(3)` and `foo(3.14)` resolve to different symbols.  


### 3.6.2 Vtable Layout  
Vtables are stored in the data section, with a fixed layout (per ABI):  
- First entry: Offset to top (for multiple inheritance).  
- Second entry: Pointer to type info (for `dynamic_cast` and `typeid`).  
- Subsequent entries: Virtual function pointers.  

**Assembly for `Car` Vtable**:  
```asm
    .section .data.rel.ro,"aw",@progbits
    .align 8
_ZTV3Car:  ; Car's vtable
    .quad 0  ; Offset to top
    .quad _ZTI3Car  ; Type info
    .quad _ZN3Car10accelerateEv  ; Car::accelerate()
```  


### 3.6.3 Exception Handling  
Exceptions use **EHFrames** (exception handling frames) to track stack unwinding. Assembly includes metadata for the linker to map exception ranges to handlers:  

```asm
    .section .gcc_except_table,"a",@progbits
    ; Exception table: maps try ranges to catch handlers
    .byte 0x1  ; Version
    .byte 0x0  ; Padding
    .uleb128 0x1  ; Number of entries
```  


## 3.7 Backward Compatibility: Compiling C Code  
C++ compilers retain compatibility with C by disabling C++ features when compiling C code (via flags like `-std=c99`). This reuse of C’s infrastructure is critical:  
- The lexer ignores C++ keywords (e.g., `class` is treated as an identifier in C mode).  
- The parser rejects C++ syntax (e.g., `//` comments are allowed in C99+, but `class` declarations are not).  
- Semantic analysis skips C++ checks (e.g., no access control for struct members).  


## 3.8 Summary  
Modern C++ compilers build on C’s core stages but extend them to handle C++’s complexity:  
- **Lexer**: Recognizes C++ keywords, operators, and literals.  
- **Parser**: Constructs ASTs for classes, templates, and lambdas.  
- **Semantic Analysis**: Validates OOP rules, template constraints, and modern features.  
- **IR**: Uses LLVM IR/GIMPLE to represent vtables, exceptions, and template instantiations.  
- **Assembly**: Follows the Itanium ABI for mangling, vtables, and exceptions.  

This incremental extension ensures C++ compilers remain powerful, backward-compatible, and capable of supporting the latest C++ standards—all while reusing the foundational compiler architecture established by C.

Figure: Extension of Module 2 + C++-Specific Stages + LLVM IR
```
[C++ Source Code (classes/templates/lambdas)]  ← Input (Oval)
       ↓ "C++ code with `class Car{}`, `template <typename T>`, `[](){...}`"
┌────────────────────────────-─┐
│ Frontend: Lexer              │  ← Light Blue (Frontend)
│ (Adds C++ tokens:            │
│  - `class`, `template`, `[]` │
│  - `override`, `constexpr`)  │
└────────────────┬────────────-┘
                 ↓ "C++ Tokens"
┌─────────────────────────────----------┐
│ Frontend: Parser                      │  ← Light Blue (Frontend)
│ (C++-Specific AST Nodes:              │
│  • ClassNode (with access specifiers) │
│  • TemplateNode (type params)         │
│  • LambdaNode (captures/body))        │
└────────────────┬────────────----------┘
                 ↓ "C++ AST"
┌─────────────────────────────------------------┐
│ Semantic Analysis                             │  ← Light Yellow (Optimization)
│ (C++-Unique Checks:                           │
│  • OOP Validation (virtual overrides)         │
│  • Template Instantiation (e.g., `List<int>`) │
│  • C++20 Concepts (type constraints))         │
└────────────────┬────────────------------------┘
                 ↓ "Validated C++ AST"
┌─────────────────────────────--------┐
│ IR Generation: LLVM IR              │  ← Light Green (IR)
│ (Replaces TACKY; models:            │
│  • Vtables (virtual funcs)          │
│  • Exceptions (`llvm.eh.exception`) │
│  • Templates (type-specific IR))    │
└────────────────┬────────────--------┘
                 ↓ "C++-Optimized LLVM IR"
┌─────────────────────────────----------------------┐
│ Backend: Assembly Generator                       │  ← Light Orange (Backend)
│ (Follows Itanium C++ ABI:                         │
│  • Name Mangling (e.g., `_Z3fooi` for `foo(int)`) │
│  • Vtable Layout (data section storage))          │
└────────────────┬────────────----------------------┘
                 ↓ "C++-Compliant x64 Assembly"
[C++ Executable (e.g., `a.out`)]  ← Output (Oval)
```