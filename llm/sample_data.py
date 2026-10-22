"""Sample training data for initial proof-of-concept training.

These are simple, repetitive code snippets that the model should be able
to learn patterns from. Real programming data can be added later.
"""

SAMPLE_DATA = [
    "def hello():\n    print('Hello, World!')\n    return 'hello'\n",
    "def add(a, b):\n    return a + b\n",
    "def subtract(a, b):\n    return a - b\n",
    "def multiply(a, b):\n    return a * b\n",
    "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b\n",
    "def square(x):\n    return x * x\n",
    "def cube(x):\n    return x * x * x\n",
    "def power(base, exp):\n    result = 1\n    for i in range(exp):\n        result *= base\n    return result\n",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "def is_even(n):\n    return n % 2 == 0\n",
    "def is_odd(n):\n    return n % 2 != 0\n",
    "def max_of_two(a, b):\n    if a > b:\n        return a\n    return b\n",
    "def min_of_two(a, b):\n    if a < b:\n        return a\n    return b\n",
    "def greet(name):\n    return f'Hello, {name}!'\n",
    "def count_vowels(s):\n    count = 0\n    for c in s:\n        if c in 'aeiou':\n            count += 1\n    return count\n",
    "def reverse_string(s):\n    return s[::-1]\n",
    "def to_uppercase(s):\n    return s.upper()\n",
    "def to_lowercase(s):\n    return s.lower()\n",
    "def length(s):\n    return len(s)\n",
    "const add = (a, b) => a + b;\n",
    "const subtract = (a, b) => a - b;\n",
    "const multiply = (a, b) => a * b;\n",
    "function greet(name) {\n  return `Hello, ${name}!`;\n}\n",
    "function square(x) {\n  return x * x;\n}\n",
    "function isEven(n) {\n  return n % 2 === 0;\n}\n",
    "const fibonacci = (n) => n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2);\n",
    "import React, { useState } from 'react';\n",
    "function Counter() {\n  const [count, setCount] = useState(0);\n  return <button onClick={() => setCount(count + 1)}>Count: {count}</button>;\n}\n",
    "export default function App() {\n  return <div>Hello World</div>;\n}\n",
    "interface User {\n  id: number;\n  name: string;\n  email: string;\n}\n",
    "function getUser(id: number): User {\n  return { id, name: 'Alice', email: 'alice@example.com' };\n}\n",
    "type Status = 'active' | 'inactive' | 'pending';\n",
    "async function fetchData(url: string): Promise<Response> {\n  const res = await fetch(url);\n  return res.json();\n}\n",
    "export async function getServerSideProps() {\n  const data = await fetch('https://api.example.com/data');\n  return { props: { data: await data.json() } };\n}\n",
    "import { NextPage } from 'next';\nconst Home: NextPage = () => <h1>Hello Next.js</h1>;\nexport default Home;\n",
]
