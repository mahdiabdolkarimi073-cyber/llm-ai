"""Real programming training corpus: Python, JavaScript, TypeScript, React, Next.js.

All examples are originally written for this project (not copied from external sources).
Includes: real code, documentation snippets, examples, error fixes, and technical Q&A.
Organized as CodeDocument objects with metadata for the data pipeline.
"""

from data_pipeline import CodeDocument


def build_corpus() -> list[CodeDocument]:
    """Build the full training corpus as a list of CodeDocument objects."""
    docs: list[CodeDocument] = []

    # ===================================================================
    # PYTHON — Code
    # ===================================================================

    python_code = [
        """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

numbers = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(numbers))
""",
        """class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        node = {'value': value, 'next': None}
        if self.head is None:
            self.head = node
            return
        current = self.head
        while current['next'] is not None:
            current = current['next']
        current['next'] = node

    def to_list(self):
        result = []
        current = self.head
        while current is not None:
            result.append(current['value'])
            current = current['next']
        return result

ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.to_list())
""",
        """import json

def save_config(config, filepath):
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

config = {'host': 'localhost', 'port': 8080, 'debug': True}
save_config(config, 'config.json')
loaded = load_config('config.json')
print(loaded)
""",
        """class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError('pop from empty stack')
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError('peek from empty stack')
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)
""",
        """def memoize(func):
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@memoize
def expensive_computation(n):
    result = 0
    for i in range(n):
        result += i ** 2
    return result

print(expensive_computation(1000))
""",
        """from contextlib import contextmanager

@contextmanager
def file_manager(filepath, mode):
    f = open(filepath, mode)
    try:
        yield f
    finally:
        f.close()

with file_manager('output.txt', 'w') as f:
    f.write('Hello, World!')
""",
        """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

sorted_list = [1, 3, 5, 7, 9, 11, 13, 15]
index = binary_search(sorted_list, 7)
print(f'Found at index: {index}')
""",
        """class Observable:
    def __init__(self):
        self._observers = []

    def subscribe(self, callback):
        self._observers.append(callback)

    def notify(self, data):
        for observer in self._observers:
            observer(data)

subject = Observable()
subject.subscribe(lambda x: print(f'Observer 1: {x}'))
subject.subscribe(lambda x: print(f'Observer 2: {x}'))
subject.notify('Event occurred')
""",
        """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

print(merge_sort([5, 2, 8, 1, 9, 3]))
""",
        """import sqlite3

def create_table(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_user(db_path, name, email):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))
    conn.commit()
    conn.close()

create_table('app.db')
insert_user('app.db', 'Alice', 'alice@example.com')
""",
        """def debounce(wait):
    def decorator(func):
        last_call = {'time': 0}
        def wrapper(*args, **kwargs):
            import time
            now = time.time()
            if now - last_call['time'] >= wait:
                last_call['time'] = now
                return func(*args, **kwargs)
        return wrapper
    return decorator

@debounce(1.0)
def on_click():
    print('Button clicked')
""",
        """class Shape:
    def area(self):
        raise NotImplementedError

    def perimeter(self):
        raise NotImplementedError

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        import math
        return math.pi * self.radius ** 2

    def perimeter(self):
        import math
        return 2 * math.pi * self.radius

shapes = [Rectangle(4, 5), Circle(3)]
for shape in shapes:
    print(f'Area: {shape.area():.2f}')
""",
        """def group_by(items, key_func):
    groups = {}
    for item in items:
        key = key_func(item)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups

users = [
    {'name': 'Alice', 'dept': 'Engineering'},
    {'name': 'Bob', 'dept': 'Sales'},
    {'name': 'Charlie', 'dept': 'Engineering'},
    {'name': 'Diana', 'dept': 'Marketing'},
]
grouped = group_by(users, lambda u: u['dept'])
for dept, members in grouped.items():
    print(f'{dept}: {[m["name"] for m in members]}')
""",
        """def pipe(*functions):
    def piped(value):
        for func in functions:
            value = func(value)
        return value
    return piped

process = pipe(
    lambda x: x + 1,
    lambda x: x * 2,
    lambda x: x - 3
)
print(process(5))
""",
        """def retry(max_attempts=3, delay=1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data():
    import random
    if random.random() < 0.5:
        raise ConnectionError('Network error')
    return 'data fetched'
""",
        """def tree_dfs(node, visit):
    visit(node)
    for child in node.get('children', []):
        tree_dfs(child, visit)

tree = {
    'value': 1,
    'children': [
        {'value': 2, 'children': [{'value': 4, 'children': []}]},
        {'value': 3, 'children': [{'value': 5, 'children': []}, {'value': 6, 'children': []}]},
    ]
}
tree_dfs(tree, lambda n: print(n['value'], end=' '))
""",
        """def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

print(flatten([1, [2, [3, 4], 5], [6, 7]]))
""",
        """def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more: curried(*args, *more)
    return curried

@curry
def add(a, b, c):
    return a + b + c

print(add(1)(2)(3))
print(add(1, 2)(3))
print(add(1, 2, 3))
""",
        """class PriorityQueue:
    def __init__(self):
        self._heap = []

    def push(self, item, priority):
        self._heap.append((priority, item))
        self._heap.sort(reverse=True)

    def pop(self):
        if not self._heap:
            raise IndexError('pop from empty queue')
        return self._heap.pop(0)[1]

    def is_empty(self):
        return len(self._heap) == 0

pq = PriorityQueue()
pq.push('low', 1)
pq.push('high', 3)
pq.push('medium', 2)
while not pq.is_empty():
    print(pq.pop())
""",
        """def rate_limiter(max_calls, period):
    calls = []
    def limiter():
        import time
        now = time.time()
        calls[:] = [t for t in calls if now - t < period]
        if len(calls) >= max_calls:
            return False
        calls.append(now)
        return True
    return limiter

can_call = rate_limiter(3, 1.0)
for i in range(5):
    print(f'Call {i}: {can_call()}')
""",
    ]

    for i, code in enumerate(python_code):
        docs.append(CodeDocument(
            text=code,
            language="python",
            category="code",
            source="original",
        ))

    # ===================================================================
    # PYTHON — Documentation / Q&A / Error fixes
    # ===================================================================

    python_docs = [
        """Q: How does a decorator work in Python?
A: A decorator is a function that takes another function and extends its behavior without modifying it. It uses the @ syntax. When you apply @decorator above a function, Python calls decorator(original_func) and replaces the original function with the returned wrapper. This is useful for logging, timing, authentication, and caching.

Example:
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f'Calling {func.__name__}')
        return func(*args, **kwargs)
    return wrapper

@log_calls
def greet(name):
    return f'Hello, {name}'

The wrapper receives all arguments via *args and **kwargs, calls the original function, and returns its result. This preserves the function's interface while adding behavior.""",
        """Q: What is the difference between a list and a tuple in Python?
A: Lists are mutable (can be modified after creation) and use square brackets. Tuples are immutable (cannot be changed) and use parentheses. Tuples are faster, can be used as dictionary keys, and are preferred for fixed collections. Lists support append, extend, insert, remove, and sort. Tuples have only count and index methods. Use tuples for heterogeneous data (like coordinates) and lists for homogeneous sequences that may change.""",
        """Q: How to fix "IndexError: list index out of range"?
A: This error occurs when you access an index that does not exist in the list. Fix by checking the length before accessing:

# Wrong:
my_list = [1, 2, 3]
print(my_list[5])  # IndexError

# Fix 1: Check length
if len(my_list) > 5:
    print(my_list[5])

# Fix 2: Use try/except
try:
    print(my_list[5])
except IndexError:
    print('Index does not exist')

# Fix 3: Use slicing (returns empty instead of error)
print(my_list[5:6])  # returns []""",
        """Q: What is a context manager and when should I use one?
A: A context manager manages resources using the 'with' statement. It ensures cleanup code runs even if an exception occurs. The __enter__ method sets up the resource and __exit__ tears it down. Use context managers for files, database connections, locks, and any resource that must be cleaned up. You can also create them with @contextmanager decorator and a generator that uses yield.""",
        """Q: How do generators work and why use them?
A: Generators use 'yield' instead of 'return' to produce values lazily. They pause execution after each yield and resume when the next value is requested. This means they only compute values on demand, saving memory for large sequences. A generator expression like (x**2 for x in range(1000000)) uses almost no memory compared to a list comprehension [x**2 for x in range(1000000)] which creates all values at once. Use generators when processing large datasets or infinite sequences.""",
        """Q: How to handle circular imports in Python?
A: Circular imports happen when module A imports B and B imports A. Solutions:
1. Move the shared code to a third module C that both A and B import.
2. Import inside the function instead of at the top level.
3. Use importlib.import_module() at runtime.
4. Restructure your code so the dependency is one-directional.

# Instead of top-level import:
def process_data():
    from models import User  # Import inside function
    return User.query.all()""",
        """Q: What is the difference between __init__ and __new__?
A: __new__ creates the instance (object allocation), __init__ initializes it. __new__ is called first and returns a new instance. __init__ then receives that instance and sets its attributes. You rarely need to override __new__, except for immutable types (like str, tuple) or singletons. For most classes, just use __init__.""",
    ]

    for text in python_docs:
        docs.append(CodeDocument(
            text=text,
            language="python",
            category="qa",
            source="original",
        ))

    # ===================================================================
    # JAVASCRIPT — Code
    # ===================================================================

    js_code = [
        """function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

const handleSearch = debounce((query) => {
  console.log('Searching for:', query);
}, 300);

handleSearch('hello');
""",
        """class EventEmitter {
  constructor() {
    this.events = {};
  }

  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }

  emit(event, data) {
    const callbacks = this.events[event] || [];
    callbacks.forEach(cb => cb(data));
  }

  off(event, callback) {
    this.events[event] = (this.events[event] || []).filter(cb => cb !== callback);
  }
}

const emitter = new EventEmitter();
emitter.on('click', (e) => console.log('Clicked:', e));
emitter.emit('click', { x: 10, y: 20 });
""",
        """function deepClone(obj) {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  if (Array.isArray(obj)) {
    return obj.map(deepClone);
  }
  const cloned = {};
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      cloned[key] = deepClone(obj[key]);
    }
  }
  return cloned;
}

const original = { a: 1, b: { c: 2, d: [3, 4] } };
const copy = deepClone(original);
copy.b.c = 99;
console.log(original.b.c); // 2 (unchanged)
""",
        """function compose(...fns) {
  return function(x) {
    return fns.reduceRight((acc, fn) => fn(acc), x);
  };
}

const addOne = x => x + 1;
const double = x => x * 2;
const subtractThree = x => x - 3;

const transform = compose(subtractThree, double, addOne);
console.log(transform(5)); // (5 + 1) * 2 - 3 = 9
""",
        """function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);
    }
    return function(...more) {
      return curried.apply(this, [...args, ...more]);
    };
  };
}

const add = curry((a, b, c) => a + b + c);
console.log(add(1)(2)(3));  // 6
console.log(add(1, 2)(3));  // 6
console.log(add(1, 2, 3));  // 6
""",
        """async function fetchAll(urls) {
  const promises = urls.map(url => fetch(url).then(r => r.json()));
  const results = await Promise.all(promises);
  return results;
}

async function fetchWithRetry(url, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
}

fetchAll(['https://api.example.com/users', 'https://api.example.com/posts'])
  .then(data => console.log(data))
  .catch(err => console.error(err));
""",
        """function createStore(reducer, initialState) {
  let state = initialState;
  const listeners = [];

  function getState() {
    return state;
  }

  function dispatch(action) {
    state = reducer(state, action);
    listeners.forEach(fn => fn(state));
  }

  function subscribe(listener) {
    listeners.push(listener);
    return () => {
      const idx = listeners.indexOf(listener);
      if (idx > -1) listeners.splice(idx, 1);
    };
  }

  return { getState, dispatch, subscribe };
}

function counterReducer(state = { count: 0 }, action) {
  switch (action.type) {
    case 'INCREMENT': return { count: state.count + 1 };
    case 'DECREMENT': return { count: state.count - 1 };
    default: return state;
  }
}

const store = createStore(counterReducer, { count: 0 });
store.subscribe(state => console.log('State:', state));
store.dispatch({ type: 'INCREMENT' });
store.dispatch({ type: 'INCREMENT' });
store.dispatch({ type: 'DECREMENT' });
""",
        """function memoize(fn) {
  const cache = new Map();
  return function(...args) {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn.apply(this, args);
    cache.set(key, result);
    return result;
  };
}

const fib = memoize(function(n) {
  if (n <= 1) return n;
  return fib(n - 1) + fib(n - 2);
});

console.log(fib(40));
""",
        """function throttle(func, limit) {
  let inThrottle;
  return function(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

const onScroll = throttle(() => {
  console.log('Scroll event handled');
}, 200);

window.addEventListener('scroll', onScroll);
""",
        """function pipe(...fns) {
  return function(x) {
    return fns.reduce((acc, fn) => fn(acc), x);
  };
}

const trim = s => s.trim();
const toLower = s => s.toLowerCase();
const words = s => s.split(' ');
const count = arr => arr.length;

const wordCount = pipe(trim, toLower, words, count);
console.log(wordCount('  Hello World From JavaScript  ')); // 4
""",
        """class Queue {
  constructor() {
    this.items = [];
  }

  enqueue(item) {
    this.items.push(item);
  }

  dequeue() {
    if (this.isEmpty()) throw new Error('Queue is empty');
    return this.items.shift();
  }

  front() {
    if (this.isEmpty()) throw new Error('Queue is empty');
    return this.items[0];
  }

  isEmpty() {
    return this.items.length === 0;
  }

  size() {
    return this.items.length;
  }
}

const queue = new Queue();
queue.enqueue('first');
queue.enqueue('second');
console.log(queue.dequeue()); // first
console.log(queue.front());   // second
""",
        """function Observer() {
  const observers = [];

  return {
    subscribe(fn) {
      observers.push(fn);
      return () => {
        const idx = observers.indexOf(fn);
        if (idx > -1) observers.splice(idx, 1);
      };
    },
    notify(data) {
      observers.forEach(fn => fn(data));
    }
  };
}

const news = Observer();
const unsub = news.subscribe(msg => console.log('Subscriber 1:', msg));
news.subscribe(msg => console.log('Subscriber 2:', msg));
news.notify('Breaking news!');
unsub();
news.notify('Another update');
""",
    ]

    for code in js_code:
        docs.append(CodeDocument(
            text=code,
            language="javascript",
            category="code",
            source="original",
        ))

    # ===================================================================
    # JAVASCRIPT — Documentation / Q&A
    # ===================================================================

    js_docs = [
        """Q: What is the event loop in JavaScript?
A: The event loop is the mechanism that handles asynchronous callbacks in JavaScript. JavaScript is single-threaded, so it uses a call stack and a task queue. When async operations complete (like setTimeout, fetch, or event handlers), their callbacks are placed in the task queue. The event loop continuously checks if the call stack is empty, and if so, pushes the next callback from the queue onto the stack. Microtasks (Promise callbacks) are processed before macrotasks (setTimeout).""",
        """Q: What is the difference between var, let, and const?
A: var is function-scoped and hoisted to the top of its function. let and const are block-scoped (only exist within their {} block). let allows reassignment, const does not. However, const objects can have their properties modified — only the binding is constant. Always use const by default, let when you need reassignment, and avoid var entirely in modern code.

const x = 10;
// x = 20; // TypeError
const obj = { a: 1 };
obj.a = 2; // This works!
// obj = {}; // TypeError""",
        """Q: What are closures and how do they work?
A: A closure is a function that retains access to variables from its outer scope even after the outer function has returned. When a function is created, it captures a reference to the scope chain. This allows inner functions to access outer variables. Closures are used for data encapsulation, callbacks, and function factories.

function createCounter() {
  let count = 0;
  return {
    increment: () => ++count,
    getCount: () => count
  };
}
const counter = createCounter();
counter.increment();
console.log(counter.getCount()); // 1""",
        """Q: How to fix "TypeError: Cannot read property 'x' of undefined"?
A: This happens when you access a property on undefined or null. Fix with optional chaining (?.), nullish checks, or default values:

// Wrong:
const name = user.profile.name; // TypeError if profile is undefined

// Fix 1: Optional chaining
const name = user?.profile?.name;

// Fix 2: Logical AND
const name = user && user.profile && user.profile.name;

// Fix 3: Default value
const name = (user?.profile?.name) ?? 'Unknown';""",
        """Q: What is the difference between == and === in JavaScript?
A: == performs type coercion before comparison (converts operands to the same type). === checks both value and type without coercion. Always use === to avoid unexpected behavior.

0 == ''   // true (both coerced to 0)
0 === ''  // false (different types)
null == undefined  // true
null === undefined // false
'1' == 1   // true
'1' === 1  // false""",
    ]

    for text in js_docs:
        docs.append(CodeDocument(
            text=text,
            language="javascript",
            category="qa",
            source="original",
        ))

    # ===================================================================
    # TYPESCRIPT — Code
    # ===================================================================

    ts_code = [
        """interface User {
  id: number;
  name: string;
  email: string;
  role: 'admin' | 'user' | 'guest';
  permissions?: string[];
}

interface UserRepository {
  findById(id: number): Promise<User | null>;
  findAll(): Promise<User[]>;
  create(data: Omit<User, 'id'>): Promise<User>;
  update(id: number, data: Partial<Omit<User, 'id'>>): Promise<User>;
  delete(id: number): Promise<boolean>;
}

class InMemoryUserRepository implements UserRepository {
  private users: Map<number, User> = new Map();
  private nextId = 1;

  async findById(id: number): Promise<User | null> {
    return this.users.get(id) ?? null;
  }

  async findAll(): Promise<User[]> {
    return Array.from(this.users.values());
  }

  async create(data: Omit<User, 'id'>): Promise<User> {
    const user: User = { ...data, id: this.nextId++ };
    this.users.set(user.id, user);
    return user;
  }

  async update(id: number, data: Partial<Omit<User, 'id'>>): Promise<User> {
    const existing = this.users.get(id);
    if (!existing) throw new Error(`User ${id} not found`);
    const updated = { ...existing, ...data };
    this.users.set(id, updated);
    return updated;
  }

  async delete(id: number): Promise<boolean> {
    return this.users.delete(id);
  }
}

const repo = new InMemoryUserRepository();
repo.create({ name: 'Alice', email: 'alice@example.com', role: 'admin' });
repo.findAll().then(users => console.log(users));
""",
        """type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

async function safeFetch<T>(url: string): Promise<Result<T>> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return { success: false, error: new Error(`HTTP ${response.status}`) };
    }
    const data = await response.json() as T;
    return { success: true, data };
  } catch (error) {
    return { success: false, error: error as Error };
  }
}

interface Post {
  id: number;
  title: string;
  body: string;
}

const result = await safeFetch<Post[]>('https://api.example.com/posts');
if (result.success) {
  console.log(result.data);
} else {
  console.error(result.error.message);
}
""",
        """type Brand<T, B extends string> = T & { readonly __brand: B };

type UserId = Brand<number, 'UserId'>;
type Email = Brand<string, 'Email'>;

function createUserId(id: number): UserId {
  if (id < 0) throw new Error('ID must be positive');
  return id as UserId;
}

function createEmail(email: string): Email {
  if (!email.includes('@')) throw new Error('Invalid email');
  return email as Email;
}

function getUser(id: UserId, email: Email): string {
  return `User ${id} has email ${email}`;
}

const uid = createUserId(42);
const email = createEmail('alice@example.com');
console.log(getUser(uid, email));
// getUser(42, 'alice@example.com'); // Type error! Must use branded types
""",
        """enum LogLevel {
  Debug = 'DEBUG',
  Info = 'INFO',
  Warn = 'WARN',
  Error = 'ERROR',
}

interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: Date;
  context?: Record<string, unknown>;
}

class Logger {
  private logs: LogEntry[] = [];
  private minLevel: LogLevel;

  constructor(minLevel: LogLevel = LogLevel.Info) {
    this.minLevel = minLevel;
  }

  private log(level: LogLevel, message: string, context?: Record<string, unknown>) {
    const entry: LogEntry = { level, message, timestamp: new Date(), context };
    this.logs.push(entry);
    if (level >= this.minLevel) {
      console.log(`[${level}] ${message}`, context ?? '');
    }
  }

  debug(message: string, context?: Record<string, unknown>) {
    this.log(LogLevel.Debug, message, context);
  }

  info(message: string, context?: Record<string, unknown>) {
    this.log(LogLevel.Info, message, context);
  }

  warn(message: string, context?: Record<string, unknown>) {
    this.log(LogLevel.Warn, message, context);
  }

  error(message: string, context?: Record<string, unknown>) {
    this.log(LogLevel.Error, message, context);
  }

  getLogs(): readonly LogEntry[] {
    return this.logs;
  }
}

const logger = new Logger(LogLevel.Info);
logger.info('Server started', { port: 3000 });
logger.error('Database connection failed', { host: 'localhost' });
""",
        """type EventHandler<T extends Record<string, unknown>> = (
  event: string,
  payload: T
) => void;

interface EventBus {
  on<T extends Record<string, unknown>>(
    event: string,
    handler: EventHandler<T>
  ): () => void;
  emit<T extends Record<string, unknown>>(
    event: string,
    payload: T
  ): void;
}

function createEventBus(): EventBus {
  const handlers = new Map<string, Set<EventHandler<Record<string, unknown>>>>();

  return {
    on(event, handler) {
      if (!handlers.has(event)) {
        handlers.set(event, new Set());
      }
      handlers.get(event)!.add(handler as EventHandler<Record<string, unknown>>);
      return () => handlers.get(event)?.delete(handler as EventHandler<Record<string, unknown>>);
    },
    emit(event, payload) {
      handlers.get(event)?.forEach(h => h(event, payload));
    }
  };
}

const bus = createEventBus();
const unsub = bus.on<{ userId: number }>('login', (event, payload) => {
  console.log(`${event}: user ${payload.userId} logged in`);
});
bus.emit('login', { userId: 42 });
unsub();
""",
        """interface Validator<T> {
  validate(value: unknown): value is T;
  parse(value: unknown): T;
}

function stringValidator(): Validator<string> {
  return {
    validate(value): value is string {
      return typeof value === 'string';
    },
    parse(value): string {
      if (!this.validate(value)) throw new Error('Expected string');
      return value;
    }
  };
}

function numberValidator(): Validator<number> {
  return {
    validate(value): value is number {
      return typeof value === 'number' && !isNaN(value);
    },
    parse(value): number {
      if (!this.validate(value)) throw new Error('Expected number');
      return value;
    }
  };
}

function objectValidator<T extends Record<string, unknown>>(
  shape: { [K in keyof T]: Validator<T[K]> }
): Validator<T> {
  return {
    validate(value): value is T {
      if (typeof value !== 'object' || value === null) return false;
      const obj = value as Record<string, unknown>;
      for (const key in shape) {
        if (!shape[key].validate(obj[key])) return false;
      }
      return true;
    },
    parse(value): T {
      if (!this.validate(value)) throw new Error('Object validation failed');
      return value as T;
    }
  };
}

const userValidator = objectValidator({
  name: stringValidator(),
  age: numberValidator(),
});

const parsed = userValidator.parse({ name: 'Alice', age: 30 });
console.log(parsed);
""",
        """type Result<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

function parseIntSafe(s: string): Result<number> {
  const n = parseInt(s, 10);
  if (isNaN(n)) return { ok: false, error: `Cannot parse "${s}" as integer` };
  return { ok: true, value: n };
}

function divide(a: number, b: number): Result<number> {
  if (b === 0) return { ok: false, error: 'Division by zero' };
  return { ok: true, value: a / b };
}

function chain<T, U>(
  result: Result<T>,
  fn: (value: T) => Result<U>
): Result<U> {
  if (!result.ok) return { ok: false, error: result.error };
  return fn(result.value);
}

const result = chain(parseIntSafe('10'), (n) => divide(n, 2));
if (result.ok) {
  console.log('Result:', result.value);
} else {
  console.error('Error:', result.error);
}
""",
    ]

    for code in ts_code:
        docs.append(CodeDocument(
            text=code,
            language="typescript",
            category="code",
            source="original",
        ))

    # ===================================================================
    # TYPESCRIPT — Q&A / Error fixes
    # ===================================================================

    ts_docs = [
        """Q: What is the difference between interface and type in TypeScript?
A: Both define object shapes. Interfaces use 'interface' keyword and support declaration merging (you can declare the same interface multiple times and they combine). Types use 'type' keyword and support unions, intersections, and conditional types. Interfaces are preferred for object shapes that might be extended. Types are preferred for unions, tuples, and mapped types. They are largely interchangeable for simple object definitions.

// Interface — can be re-opened
interface Window { title: string }
interface Window { width: number }

// Type — cannot be re-opened, supports unions
type Status = 'active' | 'inactive'
type Pair<T> = [T, T]""",
        """Q: How to fix "Type 'undefined' is not assignable to type 'string'"?
A: This error occurs when TypeScript expects a string but the value could be undefined. Fix by:

1. Use non-null assertion (if you are sure it is defined):
   const name: string = user.name!;

2. Provide a default value:
   const name: string = user.name ?? 'Unknown';

3. Use a type guard:
   if (user.name !== undefined) {
     const name: string = user.name;
   }

4. Make the type explicitly optional:
   interface User { name?: string }

5. Use required() in utility types:
   type RequiredUser = Required<User>;""",
        """Q: What are generics in TypeScript and how to use them?
A: Generics allow you to write reusable code that works with multiple types. They are like type variables. Use them when the specific type matters but should be flexible.

function identity<T>(value: T): T {
  return value;
}

const str = identity<string>('hello');
const num = identity<number>(42);

// With constraints
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const user = { name: 'Alice', age: 30 };
const name = getProperty(user, 'name'); // type: string
const age = getProperty(user, 'age');   // type: number""",
        """Q: How to fix "Property does not exist on type" error?
A: This occurs when you access a property that TypeScript does not know about. Solutions:

1. Extend the interface:
   interface Window { myCustomProp: string }

2. Use type assertion:
   const value = (someObj as any).unknownProp;

3. Use index signatures:
   interface DynamicObj { [key: string]: unknown }
   const obj: DynamicObj = { foo: 1 };

4. Use 'in' operator as a type guard:
   if ('customProp' in obj) { obj.customProp; }""",
        """Q: What is the 'unknown' type and how is it different from 'any'?
A: 'unknown' is a type-safe counterpart to 'any'. Both accept any value, but 'unknown' requires you to check the type before using it. 'any' disables type checking entirely. Always prefer 'unknown' over 'any' for safer code.

// any — no checks needed (unsafe)
function processAny(value: any) {
  return value.toUpperCase(); // no error, but crashes at runtime if value is number
}

// unknown — must check type first (safe)
function processUnknown(value: unknown) {
  if (typeof value === 'string') {
    return value.toUpperCase(); // TypeScript knows it is string here
  }
  throw new Error('Expected string');
}""",
    ]

    for text in ts_docs:
        docs.append(CodeDocument(
            text=text,
            language="typescript",
            category="qa",
            source="original",
        ))

    # ===================================================================
    # REACT — Code
    # ===================================================================

    react_code = [
        """import { useState, useEffect, useCallback } from 'react';

function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { data, loading, error, refetch };
}

export default useFetch;
""",
        """import { useReducer, useContext, createContext } from 'react';

const CartContext = createContext(null);

function cartReducer(state, action) {
  switch (action.type) {
    case 'ADD_ITEM':
      return {
        ...state,
        items: [...state.items, action.item],
        total: state.total + action.item.price,
      };
    case 'REMOVE_ITEM':
      return {
        ...state,
        items: state.items.filter((_, i) => i !== action.index),
        total: state.total - state.items[action.index]?.price || 0,
      };
    case 'CLEAR':
      return { items: [], total: 0 };
    default:
      return state;
  }
}

export function CartProvider({ children }) {
  const [state, dispatch] = useReducer(cartReducer, { items: [], total: 0 });

  return (
    <CartContext.Provider value={{ state, dispatch }}>
      {children}
    </CartContext.Provider>
  );
}

export function useCart() {
  const context = useContext(CartContext);
  if (!context) throw new Error('useCart must be used within CartProvider');
  return context;
}
""",
        """import { useState, useRef, useEffect } from 'react';

function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

function SearchInput() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    if (debouncedQuery) {
      console.log('Fetching results for:', debouncedQuery);
    }
  }, [debouncedQuery]);

  return (
    <input
      type="text"
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      placeholder="Search..."
    />
  );
}

export default SearchInput;
""",
        """import { useState, useCallback } from 'react';

function TodoApp() {
  const [todos, setTodos] = useState([]);
  const [input, setInput] = useState('');

  const addTodo = useCallback(() => {
    if (!input.trim()) return;
    setTodos(prev => [...prev, { id: Date.now(), text: input, done: false }]);
    setInput('');
  }, [input]);

  const toggleTodo = useCallback((id) => {
    setTodos(prev =>
      prev.map(todo => todo.id === id ? { ...todo, done: !todo.done } : todo)
    );
  }, []);

  const deleteTodo = useCallback((id) => {
    setTodos(prev => prev.filter(todo => todo.id !== id));
  }, []);

  return (
    <div>
      <h1>Todo List</h1>
      <input value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={addTodo}>Add</button>
      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            <span
              style={{ textDecoration: todo.done ? 'line-through' : 'none' }}
              onClick={() => toggleTodo(todo.id)}
            >
              {todo.text}
            </span>
            <button onClick={() => deleteTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default TodoApp;
""",
        """import { useState, useEffect, useRef } from 'react';

function useIntersectionObserver(options = {}) {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new IntersectionObserver(([entry]) => {
      setIsVisible(entry.isIntersecting);
    }, options);

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, isVisible];
}

function LazyImage({ src, alt }) {
  const [ref, isVisible] = useIntersectionObserver({ threshold: 0.1 });

  return (
    <div ref={ref} style={{ minHeight: '200px' }}>
      {isVisible ? <img src={src} alt={alt} /> : <p>Loading...</p>}
    </div>
  );
}

export default LazyImage;
""",
        """import { useState, useMemo } from 'react';

function DataTable({ data }) {
  const [sortKey, setSortKey] = useState('name');
  const [sortDir, setSortDir] = useState('asc');
  const [filter, setFilter] = useState('');

  const processedData = useMemo(() => {
    let result = [...data];

    if (filter) {
      result = result.filter(item =>
        Object.values(item).some(val =>
          String(val).toLowerCase().includes(filter.toLowerCase())
        )
      );
    }

    result.sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (aVal < bVal) return sortDir === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });

    return result;
  }, [data, sortKey, sortDir, filter]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  return (
    <div>
      <input
        placeholder="Filter..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />
      <table>
        <thead>
          <tr>
            <th onClick={() => handleSort('name')}>Name</th>
            <th onClick={() => handleSort('age')}>Age</th>
          </tr>
        </thead>
        <tbody>
          {processedData.map((row, i) => (
            <tr key={i}>
              <td>{row.name}</td>
              <td>{row.age}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default DataTable;
""",
        """import { createContext, useContext, useState } from 'react';

const ThemeContext = createContext();

function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

function ThemedButton() {
  const { theme, toggleTheme } = useContext(ThemeContext);

  return (
    <button
      onClick={toggleTheme}
      style={{
        background: theme === 'light' ? '#fff' : '#333',
        color: theme === 'light' ? '#333' : '#fff',
        padding: '10px 20px',
        border: '1px solid #ccc',
        cursor: 'pointer',
      }}
    >
      Toggle Theme (current: {theme})
    </button>
  );
}

function App() {
  return (
    <ThemeProvider>
      <ThemedButton />
    </ThemeProvider>
  );
}

export default App;
""",
        """import { useState, useEffect } from 'react';

function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(key);
      return stored ? JSON.parse(stored) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
      console.error('Failed to save to localStorage:', err);
    }
  }, [key, value]);

  return [value, setValue];
}

function SettingsPanel() {
  const [theme, setTheme] = useLocalStorage('theme', 'light');
  const [fontSize, setFontSize] = useLocalStorage('fontSize', 14);

  return (
    <div style={{ fontSize: `${fontSize}px` }}>
      <select value={theme} onChange={(e) => setTheme(e.target.value)}>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
      <input
        type="range"
        min="12"
        max="24"
        value={fontSize}
        onChange={(e) => setFontSize(Number(e.target.value))}
      />
    </div>
  );
}

export default SettingsPanel;
""",
    ]

    for code in react_code:
        docs.append(CodeDocument(
            text=code,
            language="react",
            category="code",
            source="original",
        ))

    # ===================================================================
    # REACT — Q&A
    # ===================================================================

    react_docs = [
        """Q: What is the difference between useState and useReducer?
A: useState is for simple state with a single value. useReducer is for complex state logic with multiple sub-values or when the next state depends on the previous one. useReducer centralizes state transitions in a reducer function, making logic easier to test and debug. Use useState for toggles, counters, and simple forms. Use useReducer for carts, multi-step forms, and state machines.

// useState — simple
const [count, setCount] = useState(0);

// useReducer — complex
const [state, dispatch] = useReducer(reducer, initialState);
dispatch({ type: 'ADD_ITEM', payload: item });""",
        """Q: When should I use useMemo and useCallback?
A: useMemo memoizes a computed value — use it when a calculation is expensive and dependencies rarely change. useCallback memoizes a function — use it when passing callbacks to optimized child components to prevent unnecessary re-renders. Do not overuse them; the memoization itself has a cost. Only use them when profiling shows a performance issue or when passing props to memoized children.

// useMemo — expensive calculation
const sorted = useMemo(() => data.sort(), [data]);

// useCallback — stable callback for child
const handleClick = useCallback(() => doSomething(id), [id]);""",
        """Q: How to fix "React Hook useEffect has a missing dependency"?
A: This warning appears when useEffect uses a variable not listed in the dependency array. Fix by either:

1. Adding the missing dependency:
   useEffect(() => { fetch(url); }, [url]); // Add url

2. Using useCallback for functions:
   const fetchData = useCallback(() => { ... }, [url]);
   useEffect(() => { fetchData(); }, [fetchData]);

3. Disabling the rule (only when intentional):
   useEffect(() => { doOnce(); }, []); // eslint-disable-next-line

Never omit dependencies that affect the effect — this causes stale closures and bugs.""",
        """Q: What is a custom hook and how to create one?
A: A custom hook is a function starting with 'use' that calls other hooks. It lets you extract reusable stateful logic. Custom hooks follow the same rules as built-in hooks: only call at top level, only call from React functions.

function useWindowSize() {
  const [size, setSize] = useState({ width: window.innerWidth, height: window.innerHeight });
  useEffect(() => {
    const handler = () => setSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, []);
  return size;
}

// Usage
function MyComponent() {
  const { width, height } = useWindowSize();
  return <p>Window: {width} x {height}</p>;
}""",
        """Q: What is the difference between controlled and uncontrolled components?
A: Controlled components have their value managed by React state — the input's value comes from state and changes update state. Uncontrolled components manage their own state via the DOM — you read values with refs. Use controlled components for forms that need validation or dynamic behavior. Use uncontrolled for simple cases where you only need the value on submit.

// Controlled
const [value, setValue] = useState('');
<input value={value} onChange={(e) => setValue(e.target.value)} />

// Uncontrolled
const inputRef = useRef();
<input ref={inputRef} />""",
    ]

    for text in react_docs:
        docs.append(CodeDocument(
            text=text,
            language="react",
            category="qa",
            source="original",
        ))

    # ===================================================================
    # NEXT.JS APP ROUTER — Code
    # ===================================================================

    nextjs_code = [
        """// app/layout.tsx
import { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'My Next.js App',
  description: 'Built with App Router',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
""",
        """// app/page.tsx
import { Suspense } from 'react'
import { getPosts } from '@/lib/api'

async function PostList() {
  const posts = await getPosts()

  return (
    <ul>
      {posts.map(post => (
        <li key={post.id}>
          <h2>{post.title}</h2>
          <p>{post.excerpt}</p>
        </li>
      ))}
    </ul>
  )
}

export default function HomePage() {
  return (
    <main>
      <h1>Blog</h1>
      <Suspense fallback={<p>Loading posts...</p>}>
        <PostList />
      </Suspense>
    </main>
  )
}
""",
        """// app/blog/[slug]/page.tsx
import { notFound } from 'next/navigation'
import { getPost, getAllSlugs } from '@/lib/api'

export async function generateStaticParams() {
  const slugs = await getAllSlugs()
  return slugs.map(slug => ({ slug }))
}

export async function generateMetadata({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug)
  if (!post) return { title: 'Post not found' }
  return {
    title: post.title,
    description: post.excerpt,
  }
}

export default async function BlogPost({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug)

  if (!post) {
    notFound()
  }

  return (
    <article>
      <h1>{post.title}</h1>
      <time dateTime={post.date}>{post.date}</time>
      <div dangerouslySetInnerHTML={{ __html: post.content }} />
    </article>
  )
}
""",
        """// app/api/users/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const limit = parseInt(searchParams.get('limit') || '10')
  const offset = parseInt(searchParams.get('offset') || '0')

  const users = await db.user.findMany({
    take: limit,
    skip: offset,
    orderBy: { createdAt: 'desc' },
  })

  return NextResponse.json({ users, total: users.length })
}

export async function POST(request: NextRequest) {
  const body = await request.json()

  if (!body.name || !body.email) {
    return NextResponse.json(
      { error: 'Name and email are required' },
      { status: 400 }
    )
  }

  const user = await db.user.create({
    data: { name: body.name, email: body.email },
  })

  return NextResponse.json(user, { status: 201 })
}
""",
        """// app/dashboard/loading.tsx
export default function Loading() {
  return (
    <div className="loading-container">
      <div className="loading-spinner" />
      <p>Loading dashboard...</p>
    </div>
  )
}

// app/dashboard/error.tsx
'use client'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="error-container">
      <h2>Something went wrong!</h2>
      <p>{error.message}</p>
      <button onClick={() => reset()}>Try again</button>
    </div>
  )
}

// app/dashboard/page.tsx
import { getDashboardData } from '@/lib/api'

export default async function DashboardPage() {
  const data = await getDashboardData()

  return (
    <main>
      <h1>Dashboard</h1>
      <div className="stats">
        <div className="stat-card">
          <h3>Total Users</h3>
          <p>{data.totalUsers}</p>
        </div>
        <div className="stat-card">
          <h3>Revenue</h3>
          <p>${data.revenue}</p>
        </div>
      </div>
    </main>
  )
}
""",
        """// app/products/page.tsx
import Link from 'next/link'
import { getProducts } from '@/lib/api'

export const dynamic = 'force-dynamic'

export default async function ProductsPage({
  searchParams,
}: {
  searchParams: { q?: string; category?: string }
}) {
  const products = await getProducts({
    query: searchParams.q,
    category: searchParams.category,
  })

  return (
    <main>
      <h1>Products</h1>
      <form>
        <input name="q" defaultValue={searchParams.q} placeholder="Search..." />
        <select name="category" defaultValue={searchParams.category}>
          <option value="">All</option>
          <option value="electronics">Electronics</option>
          <option value="clothing">Clothing</option>
        </select>
        <button type="submit">Filter</button>
      </form>
      <div className="grid">
        {products.map(product => (
          <Link href={`/products/${product.id}`} key={product.id}>
            <div className="card">
              <h3>{product.name}</h3>
              <p>${product.price}</p>
            </div>
          </Link>
        ))}
      </div>
    </main>
  )
}
""",
        """// middleware.ts
import { NextRequest, NextResponse } from 'next/server'

const PUBLIC_PATHS = ['/', '/login', '/register']

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl
  const token = request.cookies.get('auth-token')?.value

  if (PUBLIC_PATHS.includes(pathname)) {
    return NextResponse.next()
  }

  if (!token) {
    const loginUrl = new URL('/login', request.url)
    loginUrl.searchParams.set('redirect', pathname)
    return NextResponse.redirect(loginUrl)
  }

  const response = NextResponse.next()
  response.headers.set('x-pathname', pathname)
  return response
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}
""",
        """// app/blog/page.tsx
import Link from 'next/link'
import { getPosts } from '@/lib/api'

export const revalidate = 3600 // Revalidate every hour

export default async function BlogPage() {
  const posts = await getPosts()

  return (
    <main>
      <h1>Blog</h1>
      {posts.map(post => (
        <article key={post.id}>
          <Link href={`/blog/${post.slug}`}>
            <h2>{post.title}</h2>
          </Link>
          <p>{post.excerpt}</p>
          <small>{new Date(post.date).toLocaleDateString()}</small>
        </article>
      ))}
    </main>
  )
}
""",
        """// app/components/Counter.tsx
'use client'

import { useState } from 'react'

export default function Counter() {
  const [count, setCount] = useState(0)

  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount(c => c + 1)}>Increment</button>
      <button onClick={() => setCount(c => c - 1)}>Decrement</button>
      <button onClick={() => setCount(0)}>Reset</button>
    </div>
  )
}

// app/page.tsx — using the client component in a server component
import Counter from './components/Counter'

export default function Page() {
  return (
    <main>
      <h1>Welcome</h1>
      <Counter />
    </main>
  )
}
""",
        """// app/(auth)/login/page.tsx
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.error || 'Login failed')
      }

      router.push('/dashboard')
      router.refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <h1>Log In</h1>
      {error && <p className="error">{error}</p>}
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
        required
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
        required
      />
      <button type="submit">Log In</button>
    </form>
  )
}
""",
        """// app/api/auth/route.ts
import { NextRequest, NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { verifyPassword, createToken } from '@/lib/auth'

export async function POST(request: NextRequest) {
  const { email, password } = await request.json()

  const user = await verifyPassword(email, password)
  if (!user) {
    return NextResponse.json(
      { error: 'Invalid credentials' },
      { status: 401 }
    )
  }

  const token = await createToken({ userId: user.id })
  cookies().set('auth-token', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7,
  })

  return NextResponse.json({ success: true })
}

export async function DELETE() {
  cookies().delete('auth-token')
  return NextResponse.json({ success: true })
}
""",
        """// app/settings/page.tsx
import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'

export default async function SettingsPage() {
  const token = cookies().get('auth-token')

  if (!token) {
    redirect('/login')
  }

  const user = await getCurrentUser(token.value)

  return (
    <main>
      <h1>Settings</h1>
      <form action="/api/settings" method="POST">
        <label>
          Name
          <input name="name" defaultValue={user.name} />
        </label>
        <label>
          Email
          <input name="email" type="email" defaultValue={user.email} />
        </label>
        <button type="submit">Save</button>
      </form>
    </main>
  )
}
""",
        """// lib/db.ts — Database utility for Next.js App Router
import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as { prisma: PrismaClient }

export const db = globalForPrisma.prisma || new PrismaClient()

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = db
}

// Usage in server components:
// import { db } from '@/lib/db'
// const users = await db.user.findMany()
""",
    ]

    for code in nextjs_code:
        docs.append(CodeDocument(
            text=code,
            language="nextjs",
            category="code",
            source="original",
        ))

    # ===================================================================
    # NEXT.JS — Q&A / Error fixes
    # ===================================================================

    nextjs_docs = [
        """Q: What is the difference between Pages Router and App Router in Next.js?
A: The Pages Router uses the /pages directory with files like pages/index.tsx. The App Router uses the /app directory with file-based routing using page.tsx, layout.tsx, loading.tsx, and error.tsx. App Router supports Server Components by default, nested layouts, streaming with Suspense, and server-side data fetching without getServerSideProps. App Router is the recommended approach for new projects (Next.js 13+). Pages Router is still supported but receives fewer new features.

App Router structure:
app/
  layout.tsx    — shared layout
  page.tsx      — home page
  blog/
    page.tsx    — blog list
    [slug]/
      page.tsx  — dynamic route""",
        """Q: How to fix "use client" directive error in Next.js App Router?
A: If you use hooks (useState, useEffect, useRouter) or event handlers (onClick, onChange) in a component, it must be a Client Component. Add 'use client' at the top of the file:

'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'

export default function MyComponent() {
  const [count, setCount] = useState(0)
  const router = useRouter()
  // ...
}

Server Components (default in App Router) cannot use hooks, event handlers, or browser APIs. They CAN use async/await for data fetching.""",
        """Q: How to do data fetching in Next.js App Router?
A: In App Router, Server Components can fetch data directly using async/await. No need for getServerSideProps or getStaticProps.

// Server Component — fetch at request time
async function getData() {
  const res = await fetch('https://api.example.com/data')
  return res.json()
}

export default async function Page() {
  const data = await getData()
  return <div>{data.title}</div>
}

// With caching control
export const revalidate = 3600 // ISR — revalidate every hour
// or
export const dynamic = 'force-dynamic' // Always fetch fresh

// With search params
export default async function Page({ searchParams }) {
  const data = await getData(searchParams.q)
  // ...
}""",
        """Q: How to fix "Error: Invariant: headers() was used outside a request scope"?
A: This error occurs when you call Next.js request-scoped functions (headers(), cookies()) outside of a request context. Fix by ensuring they are called inside a server component or route handler that runs during a request:

// Correct — inside a server component
import { cookies } from 'next/headers'

export default async function Page() {
  const token = cookies().get('auth-token')
  // ...
}

// Wrong — inside a utility function called at build time
// Do not call cookies() in a regular function that is not
// invoked during request handling.""",
        """Q: How to create a loading state in Next.js App Router?
A: Create a loading.tsx file in any route segment. Next.js wraps the page in a Suspense boundary automatically:

// app/dashboard/loading.tsx
export default function Loading() {
  return <p>Loading...</p>
}

// app/dashboard/error.tsx — error boundary
'use client'
export default function Error({ error, reset }) {
  return (
    <div>
      <p>{error.message}</p>
      <button onClick={reset}>Try again</button>
    </div>
  )
}

// The file structure:
// app/dashboard/
//   page.tsx      — the page content
//   loading.tsx   — shown while page loads
//   error.tsx     — shown if page throws

Loading and error states are scoped to their directory. Nested routes inherit parent loading states.""",
        """Q: How to use middleware in Next.js App Router?
A: Create a middleware.ts file at the root (or src/). It runs before every matched route:

import { NextRequest, NextResponse } from 'next/server'

export function middleware(request: NextRequest) {
  const token = request.cookies.get('token')
  if (!token && !request.nextUrl.pathname.startsWith('/login')) {
    return NextResponse.redirect(new URL('/login', request.url))
  }
  return NextResponse.next()
}

export const config = {
  matcher: ['/dashboard/:path*', '/api/:path*'],
}

Middleware runs on the Edge runtime. Avoid heavy computation. Use it for auth checks, redirects, and header manipulation.""",
        """Q: How to handle dynamic routes in Next.js App Router?
A: Use bracket notation in folder names. [slug] creates a dynamic segment. [id] is another example:

// app/blog/[slug]/page.tsx
export default async function Page({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug)
  return <article>{post.title}</article>
}

// Catch-all routes: [...slug]
// app/shop/[...slug]/page.tsx
export default function Page({ params }: { params: { slug: string[] } }) {
  // /shop/a/b/c -> params.slug = ['a', 'b', 'c']
}

// For static generation of dynamic routes:
export async function generateStaticParams() {
  const posts = await getAllPosts()
  return posts.map(post => ({ slug: post.slug }))
}""",
    ]

    for text in nextjs_docs:
        docs.append(CodeDocument(
            text=text,
            language="nextjs",
            category="qa",
            source="original",
        ))

    return docs


if __name__ == "__main__":
    corpus = build_corpus()
    print(f"Total documents: {len(corpus)}")
    from collections import Counter
    lang_counts = Counter(d.language for d in corpus)
    cat_counts = Counter(d.category for d in corpus)
    print(f"Languages: {dict(lang_counts)}")
    print(f"Categories: {dict(cat_counts)}")
    print(f"Total chars: {sum(len(d.text) for d in corpus):,}")
