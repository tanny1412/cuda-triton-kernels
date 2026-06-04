# Session P3 Devlog — Memory & Pointers Lab

## Why this session
Structs and pointer-to-struct are used heavily in CUDA kernel code.
This session ties together everything from P1 and P2.

## Structs
- A struct groups related variables under one name
- No methods, just data (like a Python class with only attributes)
- Without structs: every function needs many separate arguments
- With structs: pass one clean object instead

## Accessing struct fields
- m.rows    → m is a struct directly, use dot (.)
- p->rows   → p is a POINTER to a struct, use arrow (->)
- p->rows is shorthand for (*p).rows — dereference then access

## Still to cover
- Writing and running memory_lab.cpp with structs
