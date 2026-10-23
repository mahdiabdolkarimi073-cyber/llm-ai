"""Persian conversation + Next.js/TypeScript code training corpus.

This module provides the data structure for Persian chat conversations
and Next.js code generation examples. The actual conversation data will
be added manually in CONVERSATIONS and CODE_EXAMPLES below.

Each conversation is a list of turns. Each turn has:
  - role: "user" or "assistant"
  - text: the message content (Persian for chat, code for programming)

The serialization format uses special tokens:
  <|user|>message text<|end|><|assistant|>response text<|end|>

This format enables:
  - Loss masking: only assistant tokens contribute to loss
  - Stop generation at <|end|> during inference
  - Clear role separation for the model
"""

from data_pipeline import CodeDocument


# Special tokens used in conversation format
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"

SPECIAL_TOKENS: dict[str, int] = {}
# IDs assigned after BPE training; see train_conversation_fa.py


# ---------------------------------------------------------------------------
# Conversation data — to be filled in manually
# ---------------------------------------------------------------------------

# Each entry is a dict with:
#   "turns": list of {"role": "user"|"assistant", "text": str}
#   "category": "chat" | "code_generation" | "explanation"
#   "language": "fa" | "fa_code"


CONVERSATIONS: list[dict] = [
    # 1
    {
        "category": "chat", "language": "fa",
        "turns": [
            {"role": "user", "text": "سلام"},
            {"role": "assistant", "text": "سلام! خوش آمدی. چطور می‌توانم کمکت کنم؟"},
        ],
    },
    # 2
    {
        "category": "chat", "language": "fa",
        "turns": [
            {"role": "user", "text": "تو چه کارهایی می‌توانی انجام بدهی؟"},
            {"role": "assistant", "text": "می‌توانم به پرسش‌ها پاسخ بدهم، مفاهیم را توضیح بدهم، در نوشتن و بررسی کد کمک کنم و برای حل خطاها راهکار پیشنهاد بدهم. کیفیت پاسخ‌ها به دانش و توانایی‌های مدل بستگی دارد."},
        ],
    },
    # 3
    {
        "category": "chat", "language": "fa",
        "turns": [
            {"role": "user", "text": "اگر جواب سؤالی را ندانی چه می‌کنی؟"},
            {"role": "assistant", "text": "نباید پاسخ را حدس بزنم و آن را حقیقت قطعی معرفی کنم. باید مشخص کنم کدام بخش را نمی‌دانم و در صورت نیاز اطلاعات بیشتری درخواست کنم."},
        ],
    },
    # 4
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "الگوریتم چیست؟"},
            {"role": "assistant", "text": "الگوریتم مجموعه‌ای مرتب از مراحل برای حل یک مسئله است. برای مثال، الگوریتم جمع دو عدد ابتدا دو عدد را دریافت می‌کند، آن‌ها را جمع می‌کند و نتیجه را برمی‌گرداند."},
        ],
    },
    # 5
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "فرق فرانت‌اند و بک‌اند چیست؟"},
            {"role": "assistant", "text": "فرانت‌اند بخش قابل مشاهده و تعاملی برنامه است که کاربر با آن کار می‌کند. بک‌اند منطق سمت سرور، پردازش درخواست‌ها و معمولاً ارتباط با دیتابیس را مدیریت می‌کند. این دو بخش از طریق رابط‌هایی مانند API با هم ارتباط دارند."},
        ],
    },
    # 6
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "API چیست؟"},
            {"role": "assistant", "text": "API قراردادی برای ارتباط بین نرم‌افزارهاست. برای مثال، فرانت‌اند می‌تواند با ارسال درخواست HTTP به یک API اطلاعات محصولات را از سرور دریافت کند."},
        ],
    },
    # 7
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "دیتابیس چه کاربردی دارد؟"},
            {"role": "assistant", "text": "دیتابیس برای ذخیره، سازمان‌دهی و بازیابی اطلاعات استفاده می‌شود. یک فروشگاه اینترنتی می‌تواند اطلاعات کاربران، محصولات، سفارش‌ها و پرداخت‌ها را در دیتابیس نگهداری کند."},
        ],
    },
    # 8
    {
        "category": "chat", "language": "fa",
        "turns": [
            {"role": "user", "text": "چطور یک مسئله برنامه‌نویسی را حل کنم؟"},
            {"role": "assistant", "text": "ابتدا مسئله و خروجی مورد انتظار را مشخص کن. سپس آن را به مراحل کوچک‌تر تقسیم کن، راه‌حل را پیاده‌سازی کن و با ورودی‌های عادی، مرزی و نامعتبر آزمایش کن. در صورت بروز خطا، پیام خطا و محل وقوع آن را بررسی کن."},
        ],
    },
    # 9
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "متغیر چیست؟"},
            {"role": "assistant", "text": "متغیر نامی برای دسترسی به یک مقدار در برنامه است. برای مثال، در JavaScript عبارت const age = 25 یک متغیر ثابت به نام age تعریف می‌کند که مقدار اولیه آن ۲۵ است."},
        ],
    },
    # 10
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "فرق let و const در جاوااسکریپت چیست؟"},
            {"role": "assistant", "text": "با let می‌توان متغیر را دوباره مقداردهی کرد. با const نمی‌توان متغیر را دوباره به مقدار دیگری نسبت داد. با این حال، اگر مقدار const یک شیء باشد، تغییر ویژگی‌های آن شیء همچنان ممکن است."},
        ],
    },
    # 11
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک تابع جاوااسکریپت برای جمع دو عدد بنویس."},
            {"role": "assistant", "text": "function add(a, b) {\n  return a + b;\n}"},
        ],
    },
    # 12
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "تابعی بنویس که مشخص کند عدد زوج است یا نه."},
            {"role": "assistant", "text": "function isEven(number) {\n  return number % 2 === 0;\n}"},
        ],
    },
    # 13
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "چطور یک آرایه را در جاوااسکریپت پیمایش کنم؟"},
            {"role": "assistant", "text": "const fruits = [\"apple\", \"orange\", \"banana\"];\n\nfor (const fruit of fruits) {\n  console.log(fruit);\n}"},
        ],
    },
    # 14
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک تابع بنویس که بزرگ‌ترین عدد آرایه را برگرداند و آرایه خالی را هم مدیریت کند."},
            {"role": "assistant", "text": "function findMaximum(numbers) {\n  if (numbers.length === 0) {\n    return null;\n  }\n\n  return Math.max(...numbers);\n}"},
        ],
    },
    # 15
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک تابع برای معکوس کردن رشته بنویس."},
            {"role": "assistant", "text": "function reverseString(value) {\n  return [...value].reverse().join(\"\");\n}"},
        ],
    },
    # 16
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "تابع در جاوااسکریپت چیست؟"},
            {"role": "assistant", "text": "تابع بخشی از کد است که برای انجام یک کار مشخص تعریف می‌شود و می‌توان آن را بارها فراخوانی کرد. تابع می‌تواند ورودی بگیرد و با return نتیجه برگرداند."},
        ],
    },
    # 17
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک نمونه از تابع فلشی در جاوااسکریپت بنویس."},
            {"role": "assistant", "text": "const multiply = (a, b) => a * b;\n\nconsole.log(multiply(3, 4)); // 12"},
        ],
    },
    # 18
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "Promise در جاوااسکریپت چیست؟"},
            {"role": "assistant", "text": "Promise نمایانگر نتیجه یک عملیات ناهمگام است که ممکن است هنوز آماده نشده باشد. وضعیت آن می‌تواند pending، fulfilled یا rejected باشد. برای مدیریت نتیجه می‌توان از then و catch یا از async و await استفاده کرد."},
        ],
    },
    # 19
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک تابع async بنویس که JSON را از یک API دریافت کند."},
            {"role": "assistant", "text": "async function fetchData(url) {\n  const response = await fetch(url);\n\n  if (!response.ok) {\n    throw new Error(`HTTP error: ${response.status}`);\n  }\n\n  return response.json();\n}"},
        ],
    },
    # 20
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "مدیریت خطا با try و catch چگونه است؟"},
            {"role": "assistant", "text": "کد داخل try اجرا می‌شود. اگر خطایی رخ دهد، اجرای عادی آن بخش متوقف می‌شود و کنترل به catch منتقل می‌شود. در catch می‌توان خطا را ثبت کرد یا پیام مناسبی نمایش داد. خطا را بی‌دلیل نادیده نگیر."},
        ],
    },
    # 21
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "TypeScript چه تفاوتی با JavaScript دارد؟"},
            {"role": "assistant", "text": "TypeScript امکاناتی مانند نوع‌دهی ایستا و بررسی نوع هنگام توسعه را به JavaScript اضافه می‌کند. معمولاً کد TypeScript برای اجرا به JavaScript تبدیل می‌شود. این قابلیت‌ها به شناسایی بسیاری از خطاها پیش از اجرا کمک می‌کنند."},
        ],
    },
    # 22
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک تابع TypeScript برای جمع دو عدد بنویس."},
            {"role": "assistant", "text": "function add(a: number, b: number): number {\n  return a + b;\n}"},
        ],
    },
    # 23
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک type برای کاربر در TypeScript تعریف کن."},
            {"role": "assistant", "text": "type User = {\n  id: number;\n  name: string;\n  email: string;\n  isActive: boolean;\n};"},
        ],
    },
    # 24
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "interface در TypeScript چه کاربردی دارد؟"},
            {"role": "assistant", "text": "interface شکل و قرارداد یک شیء را تعریف می‌کند؛ برای مثال مشخص می‌کند یک کاربر چه ویژگی‌هایی داشته باشد. interface و type قابلیت‌های مشترک زیادی دارند، اما در بعضی موارد مانند گسترش و تعریف مجدد رفتار متفاوتی دارند."},
        ],
    },
    # 25
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک کامپوننت ساده React با TypeScript بساز که نام را نمایش دهد."},
            {"role": "assistant", "text": "type GreetingProps = {\n  name: string;\n};\n\nexport default function Greeting({ name }: GreetingProps) {\n  return <h1>سلام، {name}</h1>;\n}"},
        ],
    },
    # 26
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "React چیست؟"},
            {"role": "assistant", "text": "React کتابخانه‌ای برای ساخت رابط کاربری است. برنامه را می‌توان به کامپوننت‌های کوچک و قابل استفاده مجدد تقسیم کرد. هر کامپوننت می‌تواند بر اساس props و state محتوای رابط کاربری را نمایش دهد."},
        ],
    },
    # 27
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "props در React چیست؟"},
            {"role": "assistant", "text": "props داده‌هایی هستند که کامپوننت والد به کامپوننت فرزند می‌دهد. برای مثال، والد می‌تواند مقدار name را به یک کامپوننت Greeting ارسال کند. کامپوننت فرزند نباید props را مستقیماً تغییر دهد."},
        ],
    },
    # 28
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "state در React چیست؟"},
            {"role": "assistant", "text": "state داده‌ای است که React تغییرات آن را دنبال می‌کند و در صورت نیاز رابط کاربری را دوباره رندر می‌کند. برای مدیریت state محلی در یک کامپوننت تابعی معمولاً از useState استفاده می‌شود."},
        ],
    },
    # 29
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک دکمه شمارنده با React بساز."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useState } from \"react\";\n\nexport default function Counter() {\n  const [count, setCount] = useState(0);\n\n  return (\n    <button onClick={() => setCount((value) => value + 1)}>\n      تعداد: {count}\n    </button>\n  );\n}"},
        ],
    },
    # 30
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "useEffect در React چه زمانی استفاده می‌شود؟"},
            {"role": "assistant", "text": "useEffect برای هماهنگ‌کردن کامپوننت با سیستم‌های بیرونی مانند رویدادهای مرورگر یا اتصال‌ها استفاده می‌شود. برای محاسبات ساده‌ای که از props یا state به دست می‌آیند، معمولاً نیازی به useEffect نیست. وابستگی‌ها باید مطابق مقادیر استفاده‌شده در effect تعیین شوند."},
        ],
    },
    # 31
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "Next.js چیست؟"},
            {"role": "assistant", "text": "Next.js چارچوبی مبتنی بر React برای ساخت برنامه‌های وب است. امکاناتی مانند مسیریابی فایل‌محور، رندر سمت سرور، تولید صفحات استاتیک و Route Handler را ارائه می‌کند. قابلیت‌های در دسترس به نسخه و نوع Router پروژه بستگی دارند."},
        ],
    },
    # 32
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "در App Router نکست جی‌اس صفحه اصلی کجا قرار می‌گیرد؟"},
            {"role": "assistant", "text": "در App Router، فایل app/page.tsx معمولاً صفحه مسیر اصلی سایت را تعریف می‌کند. برای مثال، اگر پوشه app در ریشه پروژه باشد، مسیر فایل app/page.tsx به آدرس / مربوط است."},
        ],
    },
    # 33
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک صفحه اصلی ساده برای Next.js با TypeScript بنویس."},
            {"role": "assistant", "text": "export default function HomePage() {\n  return (\n    <main>\n      <h1>به وب‌سایت ما خوش آمدید</h1>\n      <p>این اولین صفحه برنامه است.</p>\n    </main>\n  );\n}"},
        ],
    },
    # 34
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "Server Component و Client Component در Next.js چه فرقی دارند؟"},
            {"role": "assistant", "text": "در App Router، کامپوننت‌ها به‌صورت پیش‌فرض Server Component هستند و می‌توانند کارهایی مانند دریافت داده در سمت سرور انجام دهند. برای قابلیت‌های تعاملی وابسته به state، رویدادهای مرورگر یا hookهای کلاینتی، از Client Component با دستور use client در ابتدای فایل استفاده می‌شود."},
        ],
    },
    # 35
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک Route Handler ساده برای GET در Next.js بنویس."},
            {"role": "assistant", "text": "import { NextResponse } from \"next/server\";\n\nexport async function GET() {\n  return NextResponse.json({ message: \"سلام\" });\n}"},
        ],
    },
    # 36
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک Route Handler برای POST بنویس که نام را از JSON دریافت کند و ورودی نامعتبر را مدیریت کند."},
            {"role": "assistant", "text": "import { NextResponse } from \"next/server\";\n\nexport async function POST(request: Request) {\n  try {\n    const body: unknown = await request.json();\n\n    if (\n      typeof body !== \"object\" ||\n      body === null ||\n      !(\"name\" in body) ||\n      typeof body.name !== \"string\" ||\n      body.name.trim() === \"\"\n    ) {\n      return NextResponse.json(\n        { error: \"نام معتبر الزامی است\" },\n        { status: 400 }\n      );\n    }\n\n    return NextResponse.json({ name: body.name.trim() });\n  } catch {\n    return NextResponse.json(\n      { error: \"بدنه درخواست معتبر نیست\" },\n      { status: 400 }\n    );\n  }\n}"},
        ],
    },
    # 37
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چطور در Next.js اطلاعات را از سرور دریافت کنم؟"},
            {"role": "assistant", "text": "در Server Component می‌توان از یک تابع async استفاده کرد و با fetch اطلاعات را دریافت کرد. پاسخ باید از نظر وضعیت HTTP و ساختار داده بررسی شود. اگر درخواست به سرویس خصوصی نیاز دارد، کلیدهای محرمانه را در کد کلاینت قرار نده."},
        ],
    },
    # 38
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک فرم ساده ایمیل در React با TypeScript بساز."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useState, type FormEvent } from \"react\";\n\nexport default function EmailForm() {\n  const [email, setEmail] = useState(\"\");\n\n  function handleSubmit(event: FormEvent<HTMLFormElement>) {\n    event.preventDefault();\n    console.log(email);\n  }\n\n  return (\n    <form onSubmit={handleSubmit}>\n      <label htmlFor=\"email\">ایمیل</label>\n      <input\n        id=\"email\"\n        type=\"email\"\n        value={email}\n        onChange={(event) => setEmail(event.target.value)}\n        required\n      />\n      <button type=\"submit\">ارسال</button>\n    </form>\n  );\n}"},
        ],
    },
    # 39
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چرا کلید API محرمانه را نباید در فرانت‌اند بگذارم؟"},
            {"role": "assistant", "text": "کد فرانت‌اند به مرورگر کاربر می‌رسد و مقادیر محرمانه‌ای که در آن قرار داده شوند ممکن است افشا شوند. درخواست‌های حساس را از طریق بک‌اند انجام بده و کلید را در متغیر محیطی سمت سرور نگهداری کن. متغیرهایی با پیشوند NEXT_PUBLIC برای مقادیر محرمانه مناسب نیستند."},
        ],
    },
    # 40
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "متغیر محیطی DATABASE_URL چیست؟"},
            {"role": "assistant", "text": "DATABASE_URL رشته اتصال برنامه به دیتابیس است و معمولاً اطلاعاتی مانند نوع دیتابیس، میزبان، پورت، نام دیتابیس و اطلاعات احراز هویت را در خود دارد. آن را در فایل محیطی محلی نگهداری کن، فایل حاوی رمز واقعی را در Git منتشر نکن و دسترسی به دیتابیس را محدود کن."},
        ],
    },
    # 41
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "PostgreSQL چیست؟"},
            {"role": "assistant", "text": "PostgreSQL یک سیستم مدیریت دیتابیس رابطه‌ای است. داده‌ها را در جدول‌ها ذخیره می‌کند و از SQL، تراکنش‌ها، محدودیت‌های داده و قابلیت‌های پیشرفته پرس‌وجو پشتیبانی می‌کند."},
        ],
    },
    # 42
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک دستور SQL برای ساخت جدول کاربران بنویس."},
            {"role": "assistant", "text": "CREATE TABLE users (\n  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n  name VARCHAR(100) NOT NULL,\n  email VARCHAR(255) NOT NULL UNIQUE,\n  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\n);"},
        ],
    },
    # 43
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چرا باید ورودی کاربر را اعتبارسنجی کنم؟"},
            {"role": "assistant", "text": "اعتبارسنجی کمک می‌کند داده‌های ناقص، نامعتبر یا خارج از محدوده وارد سیستم نشوند. اعتبارسنجی سمت کلاینت تجربه کاربری را بهتر می‌کند، اما کافی نیست؛ سرور باید ورودی‌ها را مستقل بررسی کند و برای جلوگیری از تزریق SQL از پرس‌وجوهای پارامتری استفاده شود."},
        ],
    },
    # 44
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "خطای 404 به چه معناست؟"},
            {"role": "assistant", "text": "خطای HTTP 404 یعنی منبع درخواستی پیدا نشده است. ممکن است آدرس اشتباه باشد، مسیر تعریف نشده باشد یا منبع وجود نداشته باشد. ابتدا URL، مسیرهای برنامه و آدرس درخواست را بررسی کن."},
        ],
    },
    # 45
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "خطای 500 یعنی چه؟"},
            {"role": "assistant", "text": "خطای HTTP 500 نشان می‌دهد سرور هنگام پردازش درخواست با یک خطای داخلی مواجه شده است. برای پیدا کردن علت، لاگ سرور و stack trace را بررسی کن، خطای اصلی را پیدا کن و از نمایش اطلاعات محرمانه در پاسخ عمومی جلوگیری کن."},
        ],
    },
    # 46
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "وقتی npm run dev اجرا نمی‌شود، از کجا شروع کنم؟"},
            {"role": "assistant", "text": "ابتدا متن کامل خطا را بخوان. مطمئن شو در پوشه درست پروژه هستی، package.json وجود دارد و اسکریپت dev تعریف شده است. سپس وضعیت نصب وابستگی‌ها، نسخه Node.js و اشغال‌بودن پورت را بررسی کن. بدون دیدن خطا، علت دقیق را قطعی اعلام نکن."},
        ],
    },
    # 47
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "Git چه کاربردی دارد؟"},
            {"role": "assistant", "text": "Git سیستم کنترل نسخه است و تغییرات فایل‌های پروژه را ثبت می‌کند. با آن می‌توان تاریخچه تغییرات را دید، شاخه‌های جدا ساخت و تغییرات را ادغام کرد. ثبت commitهای کوچک و معنادار پیگیری و بازگشت به نسخه‌های قبلی را آسان‌تر می‌کند."},
        ],
    },
    # 48
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک تابع پایتون بنویس که میانگین فهرستی از اعداد را حساب کند و فهرست خالی را مدیریت کند."},
            {"role": "assistant", "text": "def average(numbers):\n    if not numbers:\n        return None\n    return sum(numbers) / len(numbers)"},
        ],
    },
    # 49
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چطور کد قابل نگهداری‌تری بنویسم؟"},
            {"role": "assistant", "text": "نام‌های روشن انتخاب کن، هر تابع را روی یک مسئولیت مشخص متمرکز نگه دار، منطق تکراری را در صورت مناسب‌بودن استخراج کن و ورودی‌های نامعتبر را مدیریت کن. برای رفتارهای مهم تست بنویس و از پیچیدگی غیرضروری پرهیز کن."},
        ],
    },
    # 50
    {
        "category": "chat", "language": "fa",
        "turns": [
            {"role": "user", "text": "اگر برای حل خطای برنامه به اطلاعات بیشتری نیاز داشته باشی، چه چیزی می‌خواهی؟"},
            {"role": "assistant", "text": "متن کامل خطا، stack trace، کد مرتبط، کاری که پیش از خطا انجام شده، نسخه ابزارهای مهم و نتیجه مورد انتظار را درخواست می‌کنم. اطلاعات محرمانه مانند رمز عبور، توکن و کلید API را پیش از ارسال حذف کن."},
        ],
    },
    
    # 51 — صفحه اصلی با ساختار معنایی
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک صفحه اصلی Next.js با بخش معرفی، خدمات و فوتر بساز."},
            {"role": "assistant", "text": "export default function HomePage() {\n  return (\n    <main>\n      <section aria-labelledby=\"hero-title\">\n        <h1 id=\"hero-title\">به کسب‌وکار ما خوش آمدید</h1>\n        <p>راهکارهایی برای رشد بهتر کسب‌وکار شما.</p>\n        <a href=\"#services\">مشاهده خدمات</a>\n      </section>\n      <section id=\"services\">\n        <h2>خدمات ما</h2>\n        <p>طراحی وب، توسعه نرم‌افزار و پشتیبانی.</p>\n      </section>\n      <footer><p>تمام حقوق محفوظ است.</p></footer>\n    </main>\n  );\n}"},
        ],
    },
    # 52 — چیدمان مشترک
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک layout ساده برای App Router در Next.js بنویس."},
            {"role": "assistant", "text": "import type { Metadata } from \"next\";\n\nexport const metadata: Metadata = {\n  title: \"وب‌سایت من\",\n  description: \"وب‌سایت ساخته‌شده با Next.js\",\n};\n\nexport default function RootLayout({\n  children,\n}: Readonly<{ children: React.ReactNode }>) {\n  return (\n    <html lang=\"fa\" dir=\"rtl\">\n      <body>{children}</body>\n    </html>\n  );\n}"},
        ],
    },
    # 53 — متادیتای صفحه
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "برای صفحه محصولات Next.js متادیتای سئو تعریف کن."},
            {"role": "assistant", "text": "import type { Metadata } from \"next\";\n\nexport const metadata: Metadata = {\n  title: \"محصولات | فروشگاه من\",\n  description: \"مشاهده و بررسی محصولات فروشگاه ما.\",\n};\n\nexport default function ProductsPage() {\n  return <main><h1>محصولات</h1></main>;\n}"},
        ],
    },
    # 54 — نمایش فهرست محصولات
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "با React و TypeScript فهرست محصولات را از آرایه نمایش بده."},
            {"role": "assistant", "text": "type Product = {\n  id: number;\n  name: string;\n  price: number;\n};\n\nconst products: Product[] = [\n  { id: 1, name: \"لپ‌تاپ\", price: 25000000 },\n  { id: 2, name: \"ماوس\", price: 500000 },\n];\n\nexport default function ProductList() {\n  return (\n    <ul>\n      {products.map((product) => (\n        <li key={product.id}>\n          {product.name} — {product.price.toLocaleString(\"fa-IR\")} تومان\n        </li>\n      ))}\n    </ul>\n  );\n}"},
        ],
    },
    # 55 — جست‌وجوی محلی
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک فهرست قابل جست‌وجو در React با TypeScript بساز."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useState } from \"react\";\n\nconst items = [\"لپ‌تاپ\", \"موبایل\", \"تبلت\", \"مانیتور\"];\n\nexport default function SearchList() {\n  const [query, setQuery] = useState(\"\");\n  const filtered = items.filter((item) =>\n    item.toLocaleLowerCase(\"fa\").includes(query.toLocaleLowerCase(\"fa\"))\n  );\n\n  return (\n    <section>\n      <label htmlFor=\"search\">جست‌وجو</label>\n      <input id=\"search\" value={query} onChange={(event) => setQuery(event.target.value)} />\n      <ul>{filtered.map((item) => <li key={item}>{item}</li>)}</ul>\n      {filtered.length === 0 && <p>نتیجه‌ای پیدا نشد.</p>}\n    </section>\n  );\n}"},
        ],
    },
    # 56 — مدیریت وضعیت بارگذاری
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در React وضعیت loading و خطای دریافت اطلاعات را نمایش بده."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useEffect, useState } from \"react\";\n\ntype User = { id: number; name: string };\n\nexport default function Users() {\n  const [users, setUsers] = useState<User[]>([]);\n  const [loading, setLoading] = useState(true);\n  const [error, setError] = useState(\"\");\n\n  useEffect(() => {\n    let active = true;\n\n    async function loadUsers() {\n      try {\n        const response = await fetch(\"/api/users\");\n        if (!response.ok) throw new Error(\"دریافت اطلاعات ناموفق بود\");\n        const data: User[] = await response.json();\n        if (active) setUsers(data);\n      } catch {\n        if (active) setError(\"خطا در دریافت کاربران\");\n      } finally {\n        if (active) setLoading(false);\n      }\n    }\n\n    void loadUsers();\n    return () => { active = false; };\n  }, []);\n\n  if (loading) return <p>در حال بارگذاری...</p>;\n  if (error) return <p role=\"alert\">{error}</p>;\n  return <ul>{users.map((user) => <li key={user.id}>{user.name}</li>)}</ul>;\n}"},
        ],
    },
    # 57 — صفحه بارگذاری
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک فایل loading.tsx برای App Router بساز."},
            {"role": "assistant", "text": "export default function Loading() {\n  return (\n    <main aria-busy=\"true\" aria-live=\"polite\">\n      <p>در حال بارگذاری اطلاعات...</p>\n    </main>\n  );\n}"},
        ],
    },
    # 58 — صفحه خطا
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک error.tsx برای مدیریت خطا در Next.js بنویس."},
            {"role": "assistant", "text": "\"use client\";\n\nexport default function ErrorPage({\n  error,\n  reset,\n}: {\n  error: Error & { digest?: string };\n  reset: () => void;\n}) {\n  return (\n    <main>\n      <h2>مشکلی پیش آمده است.</h2>\n      <p>امکان بارگذاری این بخش وجود ندارد.</p>\n      <button onClick={() => reset()}>تلاش دوباره</button>\n    </main>\n  );\n}"},
        ],
    },
    # 59 — صفحه پیدا نشد
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک فایل not-found.tsx برای Next.js بساز."},
            {"role": "assistant", "text": "import Link from \"next/link\";\n\nexport default function NotFound() {\n  return (\n    <main>\n      <h1>صفحه پیدا نشد</h1>\n      <p>آدرس واردشده معتبر نیست یا صفحه حذف شده است.</p>\n      <Link href=\"/\">بازگشت به صفحه اصلی</Link>\n    </main>\n  );\n}"},
        ],
    },
    # 60 — مسیر پویا
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک صفحه محصول با مسیر پویا در App Router بنویس."},
            {"role": "assistant", "text": "type Props = {\n  params: Promise<{ id: string }>;\n};\n\nexport default async function ProductPage({ params }: Props) {\n  const { id } = await params;\n\n  return (\n    <main>\n      <h1>محصول شماره {id}</h1>\n    </main>\n  );\n}"},
        ],
    },
    # 61 — اعتبارسنجی فرم
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "فرم ثبت‌نام ساده با نام و ایمیل در React و TypeScript بساز."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useState, type FormEvent } from \"react\";\n\nexport default function SignupForm() {\n  const [name, setName] = useState(\"\");\n  const [email, setEmail] = useState(\"\");\n  const [message, setMessage] = useState(\"\");\n\n  function submit(event: FormEvent<HTMLFormElement>) {\n    event.preventDefault();\n    if (name.trim().length < 2) {\n      setMessage(\"نام باید حداقل دو نویسه داشته باشد.\");\n      return;\n    }\n    setMessage(`اطلاعات ${name.trim()} آماده ارسال است.`);\n  }\n\n  return (\n    <form onSubmit={submit}>\n      <label htmlFor=\"name\">نام</label>\n      <input id=\"name\" value={name} onChange={(e) => setName(e.target.value)} required />\n      <label htmlFor=\"email\">ایمیل</label>\n      <input id=\"email\" type=\"email\" value={email} onChange={(e) => setEmail(e.target.value)} required />\n      <button type=\"submit\">ثبت‌نام</button>\n      <p role=\"status\">{message}</p>\n    </form>\n  );\n}"},
        ],
    },
    # 62 — route برای فهرست
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک GET API در Next.js بنویس که آرایه‌ای از محصولات برگرداند."},
            {"role": "assistant", "text": "import { NextResponse } from \"next/server\";\n\nconst products = [\n  { id: 1, name: \"لپ‌تاپ\" },\n  { id: 2, name: \"موبایل\" },\n];\n\nexport async function GET() {\n  return NextResponse.json(products);\n}"},
        ],
    },
    # 63 — کد وضعیت HTTP
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "در API چه زمانی از کدهای 200، 201، 400، 401 و 404 استفاده می‌کنیم؟"},
            {"role": "assistant", "text": "کد 200 یعنی درخواست موفق بوده است. کد 201 یعنی منبع جدیدی ساخته شده است. کد 400 نشان‌دهنده درخواست نامعتبر، کد 401 نشان‌دهنده نبود احراز هویت معتبر و کد 404 به معنای پیدا نشدن منبع است. انتخاب کد درست به رفتار واقعی endpoint بستگی دارد."},
        ],
    },
    # 64 — متغیر محیطی در سرور
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چطور متغیر محیطی را در کد سمت سرور Next.js بخوانم؟"},
            {"role": "assistant", "text": "در کد سمت سرور می‌توان از process.env استفاده کرد. برای مثال const databaseUrl = process.env.DATABASE_URL مقدار متغیر را می‌خواند. اگر مقدار الزامی است، نبودن آن را صریح مدیریت کن و آن را در پاسخ کلاینت یا لاگ عمومی افشا نکن."},
        ],
    },
    # 65 — تابع بررسی متغیر الزامی
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "تابعی در TypeScript بنویس که متغیر محیطی الزامی را برگرداند و اگر وجود نداشت خطا بدهد."},
            {"role": "assistant", "text": "export function requireEnv(name: string): string {\n  const value = process.env[name];\n\n  if (!value) {\n    throw new Error(`Missing required environment variable: ${name}`);\n  }\n\n  return value;\n}"},
        ],
    },
    # 66 — پارامترهای URL
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در یک Route Handler پارامتر جست‌وجوی q را از URL بخوان."},
            {"role": "assistant", "text": "import { NextResponse } from \"next/server\";\n\nexport async function GET(request: Request) {\n  const url = new URL(request.url);\n  const query = url.searchParams.get(\"q\")?.trim() ?? \"\";\n\n  if (!query) {\n    return NextResponse.json(\n      { error: \"پارامتر q الزامی است\" },\n      { status: 400 }\n    );\n  }\n\n  return NextResponse.json({ query });\n}"},
        ],
    },
    # 67 — صفحه‌بندی
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "صفحه‌بندی یا pagination در API چیست؟"},
            {"role": "assistant", "text": "صفحه‌بندی یعنی تقسیم مجموعه بزرگ داده به بخش‌های کوچک‌تر. در روش limit و offset، limit تعداد نتایج و offset نقطه شروع را تعیین می‌کند. ورودی‌ها باید اعتبارسنجی و سقفی برای تعداد نتایج تعیین شود؛ برای مجموعه‌های بزرگ یا داده‌های پرتغییر، cursor pagination می‌تواند مناسب‌تر باشد."},
        ],
    },
    # 68 — فیلتر کردن داده‌ها
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در TypeScript محصولات را بر اساس حداقل قیمت فیلتر کن."},
            {"role": "assistant", "text": "type Product = { id: number; name: string; price: number };\n\nfunction filterByMinimumPrice(\n  products: Product[],\n  minimumPrice: number\n): Product[] {\n  return products.filter((product) => product.price >= minimumPrice);\n}"},
        ],
    },
    # 69 — مرتب‌سازی بدون تغییر آرایه اصلی
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "آرایه محصولات را بر اساس قیمت از کم به زیاد مرتب کن بدون اینکه آرایه اصلی تغییر کند."},
            {"role": "assistant", "text": "type Product = { name: string; price: number };\n\nfunction sortByPrice(products: Product[]): Product[] {\n  return [...products].sort((a, b) => a.price - b.price);\n}"},
        ],
    },
    # 70 — رفع خطای map
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "خطای Cannot read properties of undefined (reading 'map') در React یعنی چه؟"},
            {"role": "assistant", "text": "این خطا معمولاً یعنی مقداری که روی آن map فراخوانی شده undefined است. مقدار اولیه state را متناسب با داده، مثلاً آرایه خالی، تعیین کن و ساختار پاسخ API را بررسی کن. اگر داده باید آرایه باشد، نوع آن را در TypeScript مشخص کن و پیش از استفاده اعتبارسنجی انجام بده."},
        ],
    },
    # 71 — رفع خطای key
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چرا React می‌گوید هر آیتم لیست باید key داشته باشد؟"},
            {"role": "assistant", "text": "React برای تشخیص هویت آیتم‌ها هنگام رندر مجدد به key پایدار و یکتا در همان فهرست نیاز دارد. معمولاً از شناسه واقعی داده مانند item.id استفاده کن. استفاده از index برای فهرست‌هایی که مرتب یا تغییر داده می‌شوند می‌تواند باعث رفتار نادرست شود."},
        ],
    },
    # 72 — جلوگیری از درخواست تکراری
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چرا useEffect من چند بار درخواست API می‌فرستد؟"},
            {"role": "assistant", "text": "علت ممکن است تغییر وابستگی‌های effect، اجرای مجدد به‌دلیل رندر یا بررسی‌های توسعه در React Strict Mode باشد. آرایه وابستگی‌ها را بررسی کن، effect را بدون نیاز به آن بهینه نکن و برای درخواست‌ها cleanup یا لغو درخواست با AbortController را در نظر بگیر. دوبار اجراشدن در توسعه لزوماً به معنی همین رفتار در تولید نیست."},
        ],
    },
    # 73 — AbortController
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در React هنگام unmount شدن کامپوننت، درخواست fetch را لغو کن."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useEffect, useState } from \"react\";\n\nexport default function DataPanel() {\n  const [message, setMessage] = useState(\"در حال بارگذاری...\");\n\n  useEffect(() => {\n    const controller = new AbortController();\n\n    fetch(\"/api/data\", { signal: controller.signal })\n      .then((response) => {\n        if (!response.ok) throw new Error(\"Request failed\");\n        return response.json();\n      })\n      .then((data: { message: string }) => setMessage(data.message))\n      .catch((error: unknown) => {\n        if (error instanceof Error && error.name !== \"AbortError\") {\n          setMessage(\"دریافت اطلاعات ناموفق بود\");\n        }\n      });\n\n    return () => controller.abort();\n  }, []);\n\n  return <p>{message}</p>;\n}"},
        ],
    },
    # 74 — طراحی کامپوننت قابل استفاده مجدد
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک کامپوننت Button در React با نوع‌دهی TypeScript بساز که variant داشته باشد."},
            {"role": "assistant", "text": "import type { ButtonHTMLAttributes } from \"react\";\n\ntype ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {\n  variant?: \"primary\" | \"secondary\";\n};\n\nexport default function Button({\n  variant = \"primary\",\n  className = \"\",\n  ...props\n}: ButtonProps) {\n  const styles = variant === \"primary\"\n    ? \"button button-primary\"\n    : \"button button-secondary\";\n\n  return <button className={`${styles} ${className}`} {...props} />;\n}"},
        ],
    },
    # 75 — اصول دسترس‌پذیری
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "برای دسترس‌پذیری فرم‌های وب چه نکاتی را رعایت کنم؟"},
            {"role": "assistant", "text": "برای هر ورودی label مرتبط تعریف کن، از عناصر HTML معنایی استفاده کن، خطاها را با متن واضح نشان بده و امکان کار با صفحه‌کلید را حفظ کن. برای پیام‌های مهم می‌توان از aria-live یا role مناسب استفاده کرد. رنگ به‌تنهایی نباید تنها راه انتقال معنا باشد."},
        ],
    },
    # 76 — جلوگیری از SQL injection
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چطور از SQL injection جلوگیری کنم؟"},
            {"role": "assistant", "text": "از پرس‌وجوهای پارامتری یا API امن ORM استفاده کن و ورودی کاربر را به‌صورت رشته به SQL متصل نکن. اعتبارسنجی ورودی مفید است اما جایگزین پارامتری‌سازی نیست. حساب دیتابیس را با حداقل مجوزهای لازم اجرا کن."},
        ],
    },
    # 77 — تراکنش دیتابیس
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "تراکنش دیتابیس چیست و چه زمانی لازم است؟"},
            {"role": "assistant", "text": "تراکنش چند عملیات دیتابیس را به یک واحد منطقی تبدیل می‌کند. اگر همه عملیات موفق شوند، commit انجام می‌شود و اگر خطایی رخ دهد، rollback می‌تواند تغییرات را برگرداند. برای عملیاتی مانند ثبت سفارش و کاهش موجودی که باید با هم سازگار باشند، تراکنش اهمیت دارد."},
        ],
    },
    # 78 — متغیرهای TypeScript و null
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در TypeScript یک تابع بنویس که مقدار null را مدیریت کند و نام را برگرداند."},
            {"role": "assistant", "text": "function getDisplayName(name: string | null): string {\n  if (name === null || name.trim() === \"\") {\n    return \"کاربر مهمان\";\n  }\n\n  return name.trim();\n}"},
        ],
    },
    # 79 — محدود کردن تعداد نتایج
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک تابع TypeScript بنویس که تعداد نتایج را بین 1 تا 100 محدود کند."},
            {"role": "assistant", "text": "function normalizeLimit(value: number): number {\n  if (!Number.isFinite(value)) return 20;\n  return Math.min(100, Math.max(1, Math.floor(value)));\n}"},
        ],
    },
    # 80 — تشخیص مشکل پیش از تغییر کد
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "یک صفحه Next.js سفید است. آیا باید کل صفحه را از نو بنویسم؟"},
            {"role": "assistant", "text": "نه، ابتدا باید علت را پیدا کرد. کنسول مرورگر، ترمینال سرور، خطاهای شبکه، مسیر صفحه و importهای آن را بررسی کن. صفحه سفید ممکن است از خطای زمان اجرا، تنظیمات پروژه یا پاسخ ناموفق سرور باشد. بعد از مشخص‌شدن علت، کوچک‌ترین تغییر لازم را انجام بده و از بازنویسی بی‌دلیل خودداری کن."},
        ],
    },
     # 81 — گفت‌وگوی چندمرحله‌ای: تشخیص صفحه سفید
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "صفحه Next.js من سفید می‌شود. از کجا شروع کنم؟"},
            {"role": "assistant", "text": "فعلاً کد را تصادفی تغییر نده. ابتدا ترمینال سرور و Console مرورگر را بررسی کن. مشخص کن صفحه از ابتدا سفید است یا بعد از یک درخواست یا تعامل سفید می‌شود. متن اولین خطای قرمز معمولاً سرنخ اصلی است."},
            {"role": "user", "text": "در ترمینال خطایی نیست ولی Console می‌گوید Cannot read properties of undefined (reading 'map')."},
            {"role": "assistant", "text": "پس خطا احتمالاً هنگام رندر رابط کاربری رخ می‌دهد. متغیری را پیدا کن که روی آن map اجرا می‌شود و بررسی کن داده واقعاً آرایه است یا نه. اگر داده از API می‌آید، شکل پاسخ را هم بررسی کن؛ ممکن است پاسخ {\"data\": [...]} باشد ولی کد مستقیماً روی خود پاسخ map اجرا کند."},
            {"role": "user", "text": "داده از API می‌آید و پاسخ داخل data است."},
            {"role": "assistant", "text": "در این صورت به‌جای products.map، مقدار درست را از پاسخ استخراج کن و حالت اولیه را هم آرایه قرار بده. برای نمونه: const [products, setProducts] = useState<Product[]>([]); سپس بعد از بررسی پاسخ، setProducts(result.data). اگر ساختار پاسخ تضمین‌شده نیست، پیش از ثبت state نوع data را اعتبارسنجی کن. این اصلاح را انجام بده و دوباره Console را بررسی کن."},
        ],
    },

    # 82 — گفت‌وگوی چندمرحله‌ای: ساخت API محصولات
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "برای فروشگاه Next.js یک API فهرست محصولات می‌خواهم."},
            {"role": "assistant", "text": "برای App Router می‌توانی از فایل app/api/products/route.ts و تابع GET استفاده کنی. قبل از نوشتن نسخه نهایی باید بدانیم داده‌ها از دیتابیس می‌آیند یا فعلاً نمونه ثابت هستند."},
            {"role": "user", "text": "فعلاً نمونه ثابت باشد ولی جست‌وجو و صفحه‌بندی هم داشته باشد."},
            {"role": "assistant", "text": "نمونه زیر جست‌وجو با q و صفحه‌بندی با page و limit را پیاده می‌کند. در پروژه واقعی، فیلتر و صفحه‌بندی بهتر است در دیتابیس انجام شود.\n\nimport { NextResponse } from \"next/server\";\n\nconst products = [\n  { id: 1, name: \"لپ‌تاپ\", price: 500 },\n  { id: 2, name: \"موس\", price: 20 },\n  { id: 3, name: \"مانیتور\", price: 180 },\n];\n\nexport async function GET(request: Request) {\n  const url = new URL(request.url);\n  const q = url.searchParams.get(\"q\")?.trim().toLowerCase() ?? \"\";\n  const page = Math.max(1, Number(url.searchParams.get(\"page\")) || 1);\n  const rawLimit = Number(url.searchParams.get(\"limit\")) || 10;\n  const limit = Math.min(50, Math.max(1, rawLimit));\n\n  const filtered = products.filter((product) =>\n    product.name.toLowerCase().includes(q)\n  );\n  const start = (page - 1) * limit;\n\n  return NextResponse.json({\n    items: filtered.slice(start, start + limit),\n    page,\n    limit,\n    total: filtered.length,\n    totalPages: Math.ceil(filtered.length / limit),\n  });\n}"},
            {"role": "user", "text": "چرا limit را تا 50 محدود کردی؟"},
            {"role": "assistant", "text": "برای جلوگیری از درخواست‌هایی است که تعداد بسیار زیادی نتیجه می‌خواهند و می‌توانند مصرف حافظه و زمان پاسخ را بالا ببرند. مقدار 50 فقط یک سقف نمونه است؛ در پروژه واقعی باید با اندازه داده، تجربه کاربری و محدودیت‌های سرویس هماهنگ شود."},
        ],
    },

    # 83 — گفت‌وگوی چندمرحله‌ای: فرم با اعتبارسنجی
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک فرم ثبت محصول در React و TypeScript بساز."},
            {"role": "assistant", "text": "برای اینکه فرم درست طراحی شود، حداقل نام و قیمت محصول را می‌گیریم، ورودی‌ها را اعتبارسنجی می‌کنیم و ارسال را از نمایش پیام موفقیت جدا نگه می‌داریم. اگر API هنوز آماده نیست، می‌توانیم فعلاً فقط اعتبارسنجی سمت کاربر را پیاده کنیم."},
            {"role": "user", "text": "نام و قیمت کافی است و فعلاً API ندارم."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useState, type FormEvent } from \"react\";\n\ntype ProductDraft = { name: string; price: number };\n\nexport default function ProductForm() {\n  const [name, setName] = useState(\"\");\n  const [price, setPrice] = useState(\"\");\n  const [message, setMessage] = useState(\"\");\n\n  function handleSubmit(event: FormEvent<HTMLFormElement>) {\n    event.preventDefault();\n    const cleanName = name.trim();\n    const numericPrice = Number(price);\n\n    if (cleanName.length < 2) {\n      setMessage(\"نام محصول باید حداقل دو نویسه داشته باشد.\");\n      return;\n    }\n    if (price.trim() === \"\" || !Number.isFinite(numericPrice) || numericPrice < 0) {\n      setMessage(\"قیمت باید عددی معتبر و نامنفی باشد.\");\n      return;\n    }\n\n    const product: ProductDraft = { name: cleanName, price: numericPrice };\n    setMessage(`اطلاعات «${product.name}» آماده ثبت است.`);\n  }\n\n  return (\n    <form onSubmit={handleSubmit}>\n      <label htmlFor=\"product-name\">نام محصول</label>\n      <input id=\"product-name\" value={name} onChange={(event) => setName(event.target.value)} required />\n      <label htmlFor=\"product-price\">قیمت</label>\n      <input id=\"product-price\" type=\"number\" min=\"0\" step=\"0.01\" value={price} onChange={(event) => setPrice(event.target.value)} required />\n      <button type=\"submit\">بررسی اطلاعات</button>\n      <p role=\"status\">{message}</p>\n    </form>\n  );\n}"},
            {"role": "user", "text": "پس این کد محصول را در دیتابیس ذخیره نمی‌کند؟"},
            {"role": "assistant", "text": "درست است؛ فعلاً فقط ورودی‌ها را بررسی می‌کند و پیام می‌دهد. برای ذخیره واقعی، باید یک API یا Server Action و اتصال امن به دیتابیس اضافه شود. اعتبارسنجی سمت سرور هم ضروری است، چون اعتبارسنجی مرورگر به‌تنهایی قابل اعتماد نیست."},
        ],
    },

    # 84 — گفت‌وگوی چندمرحله‌ای: تشخیص علت درخواست تکراری
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "در Next.js درخواست API من گاهی دو بار اجرا می‌شود."},
            {"role": "assistant", "text": "برای تشخیص علت باید مشخص کنیم درخواست از کجا اجرا می‌شود: در useEffect سمت کلاینت، در Server Component، یا در Route Handler. همچنین تفاوت اجرای توسعه و production مهم است."},
            {"role": "user", "text": "داخل useEffect است و فقط در حالت توسعه دو بار می‌بینم."},
            {"role": "assistant", "text": "یکی از علت‌های محتمل React Strict Mode در محیط توسعه است که برای آشکار کردن برخی عوارض جانبی، چرخه effect را دوباره بررسی می‌کند. ابتدا مطمئن شو درخواست در هر رندر به‌صورت مستقیم اجرا نشده و وابستگی‌های effect درست هستند. اگر عملیات تغییردهنده داده است، آن را صرفاً به effect وابسته نکن؛ برای درخواست خواندن داده، cleanup و لغو درخواست را در نظر بگیر."},
            {"role": "user", "text": "برای لغو درخواست چه کنم؟"},
            {"role": "assistant", "text": "می‌توانی AbortController بسازی و signal را به fetch بدهی. در cleanup، controller.abort() را صدا بزن. هنگام مدیریت خطا نیز خطای لغو درخواست را از خطاهای واقعی شبکه جدا کن؛ هر خطایی را به‌عنوان شکست بارگذاری نمایش نده."},
        ],
    },

    # 85 — گفت‌وگوی چندمرحله‌ای: صفحه‌بندی بدون ورودی نامعتبر
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "تابع TypeScript برای محاسبه offset صفحه‌بندی بنویس."},
            {"role": "assistant", "text": "اگر شماره صفحه از 1 شروع شود، offset برابر است با (page - 1) * limit. بهتر است ورودی‌ها را محدود کنیم تا صفحه و تعداد نتایج صفر یا منفی نشوند."},
            {"role": "user", "text": "کد کامل تابع را بده."},
            {"role": "assistant", "text": "type Pagination = { page: number; limit: number; offset: number };\n\nexport function getPagination(pageInput: number, limitInput: number): Pagination {\n  const page = Number.isFinite(pageInput) ? Math.max(1, Math.floor(pageInput)) : 1;\n  const safeLimit = Number.isFinite(limitInput) ? Math.floor(limitInput) : 10;\n  const limit = Math.min(100, Math.max(1, safeLimit));\n\n  return { page, limit, offset: (page - 1) * limit };\n}"},
            {"role": "user", "text": "چرا Math.floor استفاده شده؟"},
            {"role": "assistant", "text": "شماره صفحه و تعداد آیتم باید عدد صحیح باشند. Math.floor بخش اعشاری را حذف می‌کند تا ورودی‌هایی مثل 2.8 به صفحه 2 تبدیل شوند. اگر ترجیح می‌دهی ورودی اعشاری نامعتبر تلقی شود، باید به‌جای تبدیل، آن را رد کنی."},
        ],
    },

    # 86 — گفت‌وگوی چندمرحله‌ای: SQL امن
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چرا نباید مقدار جست‌وجوی کاربر را مستقیم به SQL بچسبانم؟"},
            {"role": "assistant", "text": "چون ورودی می‌تواند ساختار دستور SQL را تغییر دهد و به SQL Injection منجر شود. به‌جای اتصال رشته‌ها، از پارامترهای پرس‌وجوی آماده یا API امن ORM استفاده کن."},
            {"role": "user", "text": "یک مثال با PostgreSQL و Node.js بده."},
            {"role": "assistant", "text": "با کتابخانه pg می‌توان پارامتر را جداگانه ارسال کرد:\n\nconst result = await pool.query(\n  \"SELECT id, name FROM products WHERE name ILIKE $1 LIMIT $2\",\n  [`%${searchTerm}%`, 20]\n);\n\nمقدار searchTerm در جایگاه $1 قرار می‌گیرد و به‌عنوان داده تفسیر می‌شود، نه بخشی از دستور SQL. برای نام ستون یا جهت مرتب‌سازی نمی‌توان هر مقدار دلخواهی را به همین شکل پارامتر کرد؛ آن موارد را باید از فهرست مجاز انتخاب کرد."},
        ],
    },

    # 87 — گفت‌وگوی چندمرحله‌ای: متغیر محیطی و امنیت
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "می‌توانم DATABASE_URL را با NEXT_PUBLIC_DATABASE_URL در Next.js تعریف کنم؟"},
            {"role": "assistant", "text": "خیر، برای رشته اتصال دیتابیس نباید از پیشوند NEXT_PUBLIC_ استفاده کنی. متغیرهایی با این پیشوند ممکن است در باندل سمت مرورگر قرار بگیرند. DATABASE_URL باید فقط در محیط سرور استفاده شود."},
            {"role": "user", "text": "اگر فایل env را در Git قرار بدهم چه؟"},
            {"role": "assistant", "text": "فایل حاوی اسرار را commit نکن. نام فایل‌های محیطی را در .gitignore قرار بده و یک فایل نمونه مانند .env.example بساز که فقط نام متغیرها و مقادیر ساختگی داشته باشد. اگر رمز یا کلید قبلاً وارد مخزن شده، حذف فایل به‌تنهایی کافی نیست؛ مقدار افشاشده را تعویض کن."},
        ],
    },

    # 88 — گفت‌وگوی چندمرحله‌ای: تفاوت خطاهای HTTP
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "API من برای هر خطایی status 200 برمی‌گرداند. این کار درست است؟"},
            {"role": "assistant", "text": "معمولاً نه. کد HTTP باید نتیجه درخواست را منعکس کند. پاسخ 200 برای موفقیت است؛ خطای ورودی معمولاً 400، احراز هویت نامعتبر 401، نبود مجوز 403، منبع پیدا نشده 404 و خطای پیش‌بینی‌نشده سرور 500 است."},
            {"role": "user", "text": "اگر محصول وجود نداشته باشد چه برگردانم؟"},
            {"role": "assistant", "text": "اگر endpoint برای دریافت یک محصول مشخص است و شناسه معتبر است ولی محصول پیدا نمی‌شود، 404 مناسب است. پاسخ می‌تواند JSON ساده‌ای مانند { error: \"Product not found\" } داشته باشد. از برگرداندن جزئیات داخلی دیتابیس یا stack trace به کلاینت خودداری کن."},
        ],
    },

    # 89 — گفت‌وگوی چندمرحله‌ای: طراحی حالت بارگذاری
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "برای صفحه فهرست محصولات حالت loading و error می‌خواهم."},
            {"role": "assistant", "text": "اگر داده در کامپوننت کلاینت با fetch دریافت می‌شود، سه حالت را جدا نگه دار: در حال بارگذاری، خطا و موفقیت. همچنین درخواست را در صورت unmount شدن کامپوننت لغو کن."},
            {"role": "user", "text": "نمونه کوتاه با TypeScript بده."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useEffect, useState } from \"react\";\n\ntype Product = { id: number; name: string };\n\nexport default function ProductList() {\n  const [items, setItems] = useState<Product[]>([]);\n  const [loading, setLoading] = useState(true);\n  const [error, setError] = useState<string | null>(null);\n\n  useEffect(() => {\n    const controller = new AbortController();\n\n    async function loadProducts() {\n      try {\n        const response = await fetch(\"/api/products\", { signal: controller.signal });\n        if (!response.ok) throw new Error(\"دریافت محصولات ناموفق بود.\");\n        const data: Product[] = await response.json();\n        setItems(data);\n      } catch (err) {\n        if (err instanceof Error && err.name !== \"AbortError\") {\n          setError(err.message);\n        }\n      } finally {\n        if (!controller.signal.aborted) setLoading(false);\n      }\n    }\n\n    void loadProducts();\n    return () => controller.abort();\n  }, []);\n\n  if (loading) return <p>در حال بارگذاری...</p>;\n  if (error) return <p role=\"alert\">{error}</p>;\n  if (items.length === 0) return <p>محصولی پیدا نشد.</p>;\n\n  return <ul>{items.map((item) => <li key={item.id}>{item.name}</li>)}</ul>;\n}"},
            {"role": "user", "text": "اگر API به‌جای آرایه یک شیء برگرداند چه می‌شود؟"},
            {"role": "assistant", "text": "نوع TypeScript به‌تنهایی پاسخ دریافتی از شبکه را در زمان اجرا اعتبارسنجی نمی‌کند. اگر API پاسخ { items: [...] } می‌دهد، باید از data.items استفاده کنی و بهتر است شکل پاسخ را با اعتبارسنجی زمان اجرا بررسی کنی. قرارداد فرانت‌اند و API باید یکسان باشد."},
        ],
    },

    # 90 — گفت‌وگوی چندمرحله‌ای: کد تکراری و کامپوننت قابل استفاده مجدد
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در چند صفحه دکمه‌های مشابه دارم. چطور یک Button قابل استفاده مجدد بسازم؟"},
            {"role": "assistant", "text": "یک کامپوننت بساز که ویژگی‌های استاندارد دکمه را بپذیرد و فقط موارد واقعاً مشترک را سفارشی کند. بهتر است type پیش‌فرض مشخص باشد تا دکمه داخل فرم ناخواسته فرم را ارسال نکند."},
            {"role": "user", "text": "با TypeScript و پشتیبانی از disabled بنویس."},
            {"role": "assistant", "text": "import type { ButtonHTMLAttributes, ReactNode } from \"react\";\n\ntype ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {\n  children: ReactNode;\n  variant?: \"primary\" | \"secondary\";\n};\n\nexport default function Button({\n  children,\n  variant = \"primary\",\n  type = \"button\",\n  className = \"\",\n  ...props\n}: ButtonProps) {\n  const baseClass = \"rounded px-4 py-2 font-medium disabled:cursor-not-allowed disabled:opacity-50\";\n  const variantClass = variant === \"primary\"\n    ? \"bg-blue-600 text-white\"\n    : \"border border-gray-300 bg-white text-gray-900\";\n\n  return (\n    <button\n      type={type}\n      className={`${baseClass} ${variantClass} ${className}`.trim()}\n      {...props}\n    >\n      {children}\n    </button>\n  );\n}"},
        ],
    },

    # 91 — گفت‌وگوی چندمرحله‌ای: پارامترهای مسیر پویا
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "در Next.js جدید params در صفحه من Promise است. چرا؟"},
            {"role": "assistant", "text": "در نسخه‌های جدید Next.js، بعضی APIهای پویا از جمله params در الگوی App Router به‌صورت asynchronous ارائه می‌شوند. در Server Component می‌توانی params را await کنی. اگر نسخه پروژه قدیمی‌تر باشد، امضای نوع‌ها ممکن است متفاوت باشد؛ نسخه نصب‌شده را مبنا قرار بده."},
            {"role": "user", "text": "پس برای صفحه app/products/[id]/page.tsx چه شکلی بنویسم؟"},
            {"role": "assistant", "text": "type PageProps = {\n  params: Promise<{ id: string }>;\n};\n\nexport default async function ProductPage({ params }: PageProps) {\n  const { id } = await params;\n\n  if (!id.trim()) {\n    return <main><h1>شناسه محصول معتبر نیست</h1></main>;\n  }\n\n  return <main><h1>محصول {id}</h1></main>;\n}"},
        ],
    },

    # 92 — گفت‌وگوی چندمرحله‌ای: رفع خطای import
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "خطای Module not found می‌گیرم. باید چه کنم؟"},
            {"role": "assistant", "text": "اول متن کامل خطا و مسیری را که ماژول از آن import شده بررسی کن. سپس نام فایل، بزرگی و کوچکی حروف، مسیر نسبی و نام بسته را مقایسه کن. بدون دیدن خطا، نصب دوباره همه وابستگی‌ها اولین راه‌حل مناسبی نیست."},
            {"role": "user", "text": "فایل Button.tsx است ولی من از ./button واردش کرده‌ام."},
            {"role": "assistant", "text": "اگر محیط یا ابزار ساخت به بزرگی و کوچکی حروف حساس باشد، Button و button دو نام متفاوت‌اند. مسیر import را با نام واقعی فایل یکسان کن: import Button from \"./Button\". بهتر است این هماهنگی را در همه importهای همان فایل رعایت کنی."},
        ],
    },

    # 93 — گفت‌وگوی چندمرحله‌ای: مدیریت null در TypeScript
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "TypeScript می‌گوید object is possibly null. چطور رفعش کنم؟"},
            {"role": "assistant", "text": "این هشدار می‌گوید ممکن است مقدار پیش از استفاده وجود نداشته باشد. به‌جای خاموش‌کردن بررسی نوع، شرطی اضافه کن که مقدار null را مدیریت کند یا از optional chaining استفاده کن؛ انتخاب درست به این بستگی دارد که نبود مقدار قابل قبول است یا نه."},
            {"role": "user", "text": "یک مثال برای پیدا کردن محصول با id بده."},
            {"role": "assistant", "text": "type Product = { id: number; name: string };\n\nfunction getProductName(products: Product[], id: number): string {\n  const product = products.find((item) => item.id === id);\n  if (!product) {\n    return \"محصول پیدا نشد\";\n  }\n  return product.name;\n}"},
        ],
    },

    # 94 — گفت‌وگوی چندمرحله‌ای: مرتب‌سازی امن
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چرا sort آرایه اصلی من را تغییر می‌دهد؟"},
            {"role": "assistant", "text": "متد sort آرایه موجود را درجا تغییر می‌دهد. اگر می‌خواهی state در React تغییرناپذیر بماند، ابتدا یک کپی بساز و بعد آن را مرتب کن."},
            {"role": "user", "text": "برای قیمت از کم به زیاد مثال بزن."},
            {"role": "assistant", "text": "type Product = { id: number; price: number };\n\nconst sortedProducts = [...products].sort(\n  (a, b) => a.price - b.price\n);\n\nاین کد آرایه جدیدی می‌سازد و ترتیب آرایه products را تغییر نمی‌دهد. اگر قیمت‌ها رشته باشند، ابتدا باید آن‌ها را به شکل معتبر عددی تبدیل و اعتبارسنجی کنی."},
        ],
    },

    # 95 — گفت‌وگوی چندمرحله‌ای: درخواست اطلاعات لازم برای رفع باگ
    {
        "category": "chat", "language": "fa",
        "turns": [
            {"role": "user", "text": "کدم کار نمی‌کند، درستش کن."},
            {"role": "assistant", "text": "حتماً کمک می‌کنم. برای اینکه حدس نزنم، لطفاً کد مربوط به بخش مشکل‌دار، متن کامل خطا و کاری را که انتظار داشتی در مقایسه با اتفاق واقعی بفرست."},
            {"role": "user", "text": "فقط می‌دانم بعد از کلیک روی دکمه هیچ اتفاقی نمی‌افتد."},
            {"role": "assistant", "text": "ابتدا بررسی کن دکمه داخل فرم نباشد که رفتار پیش‌فرض فرم با انتظار تو تداخل کند، و ببین handler واقعاً به onClick متصل شده است یا نه. در ابتدای handler موقتاً یک console.log بگذار. اگر اجرا نشد، مشکل در اتصال رویداد یا لایه‌ای است که کلیک را می‌گیرد؛ اگر اجرا شد، مراحل بعدی handler را بررسی می‌کنیم."},
            {"role": "user", "text": "لاگ چاپ نمی‌شود و دکمه داخل form است."},
            {"role": "assistant", "text": "اتصال onClick را بررسی کن و اگر دکمه قرار نیست فرم را ارسال کند، type=\"button\" بگذار. در HTML، دکمه داخل فرم ممکن است به‌طور پیش‌فرض submit باشد. اگر هدف ارسال فرم است، به‌جای مدیریت جداگانه کلیک، رویداد onSubmit فرم را مدیریت کن و از event.preventDefault() در صورت نیاز استفاده کن."},
        ],
    },

    # 96 — گفت‌وگوی چندمرحله‌ای: تفاوت کد سمت سرور و کلاینت
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "آیا باید بالای همه فایل‌های Next.js بنویسم use client؟"},
            {"role": "assistant", "text": "خیر. در App Router، کامپوننت‌ها به‌طور پیش‌فرض Server Component هستند. فقط وقتی به قابلیت‌های کلاینت مانند useState، useEffect یا رویدادهای تعاملی نیاز داری، مرز Client Component را با \"use client\" مشخص کن."},
            {"role": "user", "text": "پس اگر فقط اطلاعات محصول را از سرور بگیرم چه؟"},
            {"role": "assistant", "text": "اگر دریافت داده در Server Component انجام می‌شود و به state یا رویداد مرورگر نیاز نداری، معمولاً نیازی به \"use client\" نیست. این کار می‌تواند حجم JavaScript سمت مرورگر را کاهش دهد. بخش تعاملی را در صورت نیاز به یک کامپوننت کوچک کلاینتی جدا کن."},
        ],
    },

    # 97 — گفت‌وگوی چندمرحله‌ای: تست ورودی‌های مرزی
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "تابع محاسبه میانگین را نوشتم. چه تست‌هایی لازم دارد؟"},
            {"role": "assistant", "text": "فقط یک ورودی معمولی کافی نیست. ورودی عادی، آرایه خالی، یک عضو، عدد منفی، مقادیر اعشاری و ورودی نامعتبر را بررسی کن. همچنین مشخص کن تابع برای آرایه خالی باید خطا بدهد یا مقدار خاصی برگرداند."},
            {"role": "user", "text": "می‌خواهم برای آرایه خالی خطا بدهد."},
            {"role": "assistant", "text": "function average(values: number[]): number {\n  if (values.length === 0) {\n    throw new Error(\"برای محاسبه میانگین، آرایه نباید خالی باشد.\");\n  }\n  if (!values.every(Number.isFinite)) {\n    throw new Error(\"همه مقادیر باید عدد متناهی باشند.\");\n  }\n  return values.reduce((sum, value) => sum + value, 0) / values.length;\n}\n\nتست‌های مهم شامل average([4]) برابر 4، average([2, 4]) برابر 3 و پرتاب خطا برای average([]) است."},
        ],
    },

    # 98 — گفت‌وگوی چندمرحله‌ای: ساختار پروژه و پرهیز از تغییرات بی‌هدف
    {
        "category": "chat", "language": "fa",
        "turns": [
            {"role": "user", "text": "برای اضافه‌کردن یک قابلیت جدید، همه پروژه را بازنویسی کنم؟"},
            {"role": "assistant", "text": "نه؛ بازنویسی گسترده ریسک ایجاد باگ و تغییر رفتارهای سالم را بالا می‌برد. ابتدا مسیر فعلی قابلیت، قراردادهای API، مدل داده و کامپوننت‌های مرتبط را پیدا کن. سپس کوچک‌ترین تغییر سازگار را پیاده کن."},
            {"role": "user", "text": "چطور مطمئن شوم بخش‌های دیگر خراب نمی‌شوند؟"},
            {"role": "assistant", "text": "قبل از تغییر، رفتار فعلی و وابستگی‌های مستقیم را ثبت کن. تغییر را محدود نگه دار، ورودی‌ها و حالت‌های خطا را بررسی کن و تست‌های مرتبط را اجرا کن. اگر اجرای تست یا ساخت زمان‌بر یا پرهزینه است، ابتدا بررسی ایستا و تست کوچک مرتبط را انجام بده و اجرای کامل را جداگانه برنامه‌ریزی کن."},
        ],
    },

    # 99 — گفت‌وگوی چندمرحله‌ای: خروجی JSON قابل اعتماد
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در Route Handler یک پاسخ JSON خطا و موفقیت استاندارد بده."},
            {"role": "assistant", "text": "بهتر است شکل پاسخ در همه حالت‌ها روشن باشد و کد وضعیت HTTP نیز نتیجه را منعکس کند. پیام خطا نباید جزئیات حساس داخلی را فاش کند."},
            {"role": "user", "text": "یک نمونه کوتاه بده."},
            {"role": "assistant", "text": "import { NextResponse } from \"next/server\";\n\nexport async function GET() {\n  try {\n    const items: { id: number; name: string }[] = [];\n    return NextResponse.json({ ok: true, data: items });\n  } catch {\n    return NextResponse.json(\n      { ok: false, error: { code: \"INTERNAL_ERROR\", message: \"خطای داخلی سرور\" } },\n      { status: 500 }\n    );\n  }\n}"},
            {"role": "user", "text": "این catch برای خطای دیتابیس هم کافی است؟"},
            {"role": "assistant", "text": "برای نمایش پاسخ عمومی کافی است، اما در برنامه واقعی باید خطا به‌صورت امن در سمت سرور ثبت و دسته‌بندی شود. جزئیات فنی مانند رمز اتصال یا stack trace را به کلاینت نفرست. همچنین اگر عملیات ممکن است خطاهای قابل پیش‌بینی داشته باشد، آن‌ها را از خطاهای غیرمنتظره جدا کن."},
        ],
    },

    # 100 — گفت‌وگوی چندمرحله‌ای: یادگیری از خطا بدون ادعای بی‌پایه
    {
        "category": "chat", "language": "fa",
        "turns": [
            {"role": "user", "text": "این کد را اصلاح کن و تضمین بده که بدون خطاست."},
            {"role": "assistant", "text": "می‌توانم کد را بر اساس اطلاعات موجود بررسی و اصلاح پیشنهادی بدهم، اما بدون اجرای تست‌های مناسب نمی‌توانم تضمین کنم که هیچ خطایی ندارد. بهتر است فرض‌ها، تغییرات و مواردی را که هنوز بررسی نشده‌اند صریح بگویم."},
            {"role": "user", "text": "پس چطور پاسخ قابل اعتمادتری می‌دهی؟"},
            {"role": "assistant", "text": "ابتدا مسئله و رفتار مورد انتظار را مشخص می‌کنم، سپس تغییر حداقلی پیشنهاد می‌دهم، موارد مرزی و خطاهای محتمل را بررسی می‌کنم و در پایان می‌گویم چه چیزی از روی کد قابل نتیجه‌گیری است و چه چیزی به اجرای واقعی نیاز دارد. اگر اطلاعات کافی نباشد، سؤال مشخص می‌پرسم؛ نتیجه‌ای را که آزمایش نشده به‌عنوان تست موفق معرفی نمی‌کنم."},
        ],
    },
    # 101 — دریافت پارامتر مسیر پویا
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در Next.js جدید صفحه محصول با پارامتر id بنویس."},
            {"role": "assistant", "text": "در نسخه‌هایی از Next.js که params به‌صورت Promise تعریف شده است، باید آن را await کنی.\n\ntype PageProps = {\n  params: Promise<{ id: string }>;\n};\n\nexport default async function ProductPage({ params }: PageProps) {\n  const { id } = await params;\n  return <main><h1>محصول شماره {id}</h1></main>;\n}"},
        ],
    },

    # 102 — مدیریت محصول پیدا نشده
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "اگر محصول در صفحه پویای Next.js وجود نداشت، صفحه 404 نشان بده."},
            {"role": "assistant", "text": "از notFound در next/navigation استفاده کن. ابتدا محصول را از منبع داده دریافت کن و اگر وجود نداشت، notFound() را فراخوانی کن.\n\nimport { notFound } from \"next/navigation\";\n\nexport default async function ProductPage({\n  params,\n}: {\n  params: Promise<{ id: string }>;\n}) {\n  const { id } = await params;\n  const product = await getProductById(id);\n\n  if (!product) notFound();\n\n  return <main><h1>{product.name}</h1></main>;\n}"},
        ],
    },

    # 103 — ساخت API با متد POST
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک POST Route Handler در Next.js بنویس که نام محصول را دریافت کند."},
            {"role": "assistant", "text": "در Route Handler بدنه درخواست را بخوان، نوع داده را بررسی کن و ورودی نامعتبر را با وضعیت 400 رد کن.\n\nimport { NextResponse } from \"next/server\";\n\nexport async function POST(request: Request) {\n  let body: unknown;\n\n  try {\n    body = await request.json();\n  } catch {\n    return NextResponse.json(\n      { error: \"JSON نامعتبر است\" },\n      { status: 400 }\n    );\n  }\n\n  if (\n    typeof body !== \"object\" ||\n    body === null ||\n    !(\"name\" in body) ||\n    typeof body.name !== \"string\" ||\n    body.name.trim().length < 2\n  ) {\n    return NextResponse.json(\n      { error: \"نام محصول معتبر نیست\" },\n      { status: 400 }\n    );\n  }\n\n  return NextResponse.json(\n    { name: body.name.trim() },\n    { status: 201 }\n  );\n}"},
        ],
    },

    # 104 — تفاوت Server Component و Client Component
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چه زمانی در Next.js باید از use client استفاده کنم؟"},
            {"role": "assistant", "text": "کامپوننت‌های App Router به‌صورت پیش‌فرض Server Component هستند. اگر به useState، useEffect، رویدادهای مرورگر یا تعامل مستقیم کاربر نیاز داری، از دستور \"use client\" در ابتدای فایل استفاده کن. دریافت داده خصوصی و دسترسی به دیتابیس معمولاً بهتر است در سمت سرور باقی بماند."},
        ],
    },

    # 105 — ایجاد فرم تعاملی
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک فرم جست‌وجوی تعاملی در Next.js با TypeScript بساز."},
            {"role": "assistant", "text": "\"use client\";\n\nimport { useState, type FormEvent } from \"react\";\n\nexport default function SearchForm() {\n  const [query, setQuery] = useState(\"\");\n\n  function handleSubmit(event: FormEvent<HTMLFormElement>) {\n    event.preventDefault();\n    const value = query.trim();\n    if (!value) return;\n    console.log(\"Search:\", value);\n  }\n\n  return (\n    <form onSubmit={handleSubmit}>\n      <label htmlFor=\"query\">جست‌وجو</label>\n      <input\n        id=\"query\"\n        value={query}\n        onChange={(event) => setQuery(event.target.value)}\n      />\n      <button type=\"submit\">جست‌وجو</button>\n    </form>\n  );\n}"},
        ],
    },

    # 106 — دریافت داده در Server Component
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در Server Component اطلاعات محصولات را با fetch دریافت کن."},
            {"role": "assistant", "text": "کامپوننت سرور می‌تواند async باشد و مستقیماً اطلاعات را دریافت کند. پاسخ HTTP را بررسی کن و برای داده‌های حساس، سیاست کش را آگاهانه انتخاب کن.\n\ntype Product = { id: number; name: string };\n\nexport default async function ProductsPage() {\n  const response = await fetch(\"https://example.com/api/products\", {\n    cache: \"no-store\",\n  });\n\n  if (!response.ok) {\n    throw new Error(\"دریافت محصولات ناموفق بود\");\n  }\n\n  const products: Product[] = await response.json();\n\n  return (\n    <ul>\n      {products.map((product) => (\n        <li key={product.id}>{product.name}</li>\n      ))}\n    </ul>\n  );\n}"},
        ],
    },

    # 107 — مدیریت بارگذاری صفحه
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "فایل loading.tsx برای نمایش وضعیت بارگذاری در App Router بنویس."},
            {"role": "assistant", "text": "فایل loading.tsx می‌تواند رابط انتظار یک مسیر را تعریف کند و در ساختارهای مناسب، هنگام بارگذاری بخش صفحه نمایش داده شود.\n\nexport default function Loading() {\n  return (\n    <main aria-busy=\"true\" aria-live=\"polite\">\n      <p>در حال دریافت اطلاعات...</p>\n    </main>\n  );\n}"},
        ],
    },

    # 108 — مدیریت خطای صفحه
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "یک error.tsx با دکمه تلاش مجدد برای Next.js بساز."},
            {"role": "assistant", "text": "\"use client\";\n\nexport default function ErrorPage({\n  reset,\n}: {\n  error: Error & { digest?: string };\n  reset: () => void;\n}) {\n  return (\n    <main>\n      <h1>خطایی رخ داده است</h1>\n      <p>بارگذاری این بخش موفق نبود.</p>\n      <button type=\"button\" onClick={() => reset()}>\n        تلاش دوباره\n      </button>\n    </main>\n  );\n}"},
        ],
    },

    # 109 — تعریف metadata
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "برای صفحه محصولات در Next.js عنوان و توضیحات SEO تعریف کن."},
            {"role": "assistant", "text": "از Metadata API استفاده کن. عنوان و توضیحات باید دقیقاً با محتوای صفحه مرتبط باشند.\n\nimport type { Metadata } from \"next\";\n\nexport const metadata: Metadata = {\n  title: \"محصولات فروشگاه\",\n  description: \"مشاهده محصولات و بررسی مشخصات آن‌ها.\",\n};\n\nexport default function ProductsPage() {\n  return <main><h1>محصولات فروشگاه</h1></main>;\n}"},
        ],
    },

    # 110 — اتصال PostgreSQL با Prisma
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چطور Next.js را به PostgreSQL با Prisma متصل کنم؟"},
            {"role": "assistant", "text": "ابتدا provider دیتابیس را در schema.prisma روی postgresql تنظیم کن و DATABASE_URL را در متغیر محیطی سمت سرور قرار بده. سپس Prisma Client را تولید کن و یک ماژول مشترک برای دسترسی به آن بساز. اتصال دیتابیس را فقط در کد سمت سرور استفاده کن و رشته اتصال را در کد مرورگر قرار نده."},
        ],
    },

    # 111 — اعتبارسنجی با Zod
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "با Zod نام و قیمت محصول را اعتبارسنجی کن."},
            {"role": "assistant", "text": "import { z } from \"zod\";\n\nexport const productSchema = z.object({\n  name: z.string().trim().min(2).max(120),\n  price: z.number().finite().nonnegative(),\n});\n\nexport type ProductInput = z.infer<typeof productSchema>;\n\nexport function validateProduct(input: unknown) {\n  return productSchema.safeParse(input);\n}"},
        ],
    },

    # 112 — جلوگیری از SQL Injection
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چطور از SQL Injection در بک‌اند Next.js جلوگیری کنم؟"},
            {"role": "assistant", "text": "ورودی کاربر را مستقیماً به رشته SQL متصل نکن. از پارامترهای پرس‌وجو یا روش‌های امن ORM استفاده کن، ورودی را اعتبارسنجی کن و مجوزهای دیتابیس را محدود نگه دار. اعتبارسنجی به‌تنهایی جایگزین پرس‌وجوی پارامتری نیست."},
        ],
    },

    # 113 — مدیریت وضعیت HTTP
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "برای API چه تفاوتی بین وضعیت‌های 400، 401، 403، 404 و 500 وجود دارد؟"},
            {"role": "assistant", "text": "400 یعنی درخواست یا ورودی نامعتبر است. 401 یعنی احراز هویت معتبر وجود ندارد. 403 یعنی دسترسی مجاز نیست. 404 یعنی منبع پیدا نشده است. 500 نشان‌دهنده خطای داخلی پیش‌بینی‌نشده سرور است. انتخاب وضعیت باید با علت واقعی شکست هماهنگ باشد."},
        ],
    },

    # 114 — صفحه‌بندی در TypeScript
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "تابعی برای محدود کردن page و limit در صفحه‌بندی بنویس."},
            {"role": "assistant", "text": "function normalizePagination(pageValue: number, limitValue: number) {\n  const page = Number.isFinite(pageValue)\n    ? Math.max(1, Math.floor(pageValue))\n    : 1;\n\n  const requestedLimit = Number.isFinite(limitValue)\n    ? Math.floor(limitValue)\n    : 10;\n\n  const limit = Math.min(100, Math.max(1, requestedLimit));\n\n  return {\n    page,\n    limit,\n    offset: (page - 1) * limit,\n  };\n}"},
        ],
    },

    # 115 — فیلتر محصولات
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "محصولات را بر اساس نام و حداقل قیمت فیلتر کن."},
            {"role": "assistant", "text": "type Product = {\n  id: number;\n  name: string;\n  price: number;\n};\n\nfunction filterProducts(\n  products: Product[],\n  query: string,\n  minimumPrice: number\n): Product[] {\n  const normalizedQuery = query.trim().toLocaleLowerCase();\n\n  return products.filter((product) =>\n    product.name.toLocaleLowerCase().includes(normalizedQuery) &&\n    product.price >= minimumPrice\n  );\n}"},
        ],
    },

    # 116 — مدیریت خطای دریافت API
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در TypeScript تابعی بنویس که JSON را از API بگیرد و خطای HTTP را مدیریت کند."},
            {"role": "assistant", "text": "export async function fetchJson<T>(url: string): Promise<T> {\n  const response = await fetch(url);\n\n  if (!response.ok) {\n    throw new Error(`درخواست ناموفق بود: ${response.status}`);\n  }\n\n  return (await response.json()) as T;\n}"},
        ],
    },

    # 117 — مدیریت متغیر محیطی
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "چرا نباید DATABASE_URL را با NEXT_PUBLIC تعریف کنم؟"},
            {"role": "assistant", "text": "متغیرهایی که پیشوند NEXT_PUBLIC دارند ممکن است در خروجی قابل دسترس مرورگر قرار بگیرند. DATABASE_URL معمولاً حاوی اطلاعات حساس اتصال دیتابیس است و باید فقط در محیط سرور استفاده شود. اگر یک راز در مخزن عمومی منتشر شده، حذف فایل به‌تنهایی کافی نیست و باید راز را تعویض کنی."},
        ],
    },

    # 118 — تشخیص خطای صفحه سفید
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "صفحه Next.js سفید شده است. چه کار کنم؟"},
            {"role": "assistant", "text": "ابتدا Console مرورگر و ترمینال سرور را بررسی کن. اولین خطای واقعی را پیدا کن، مسیر صفحه و importهای مرتبط را بررسی کن و ببین پاسخ API موفق بوده است یا نه. قبل از بازنویسی صفحه، علت را با یک آزمایش کوچک مشخص کن. پس از اصلاح، همان مسیر و حالت خطا را دوباره آزمایش کن."},
        ],
    },

    # 119 — امنیت احراز هویت
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "آیا مخفی‌کردن دکمه مدیریت برای امنیت کافی است؟"},
            {"role": "assistant", "text": "خیر. مخفی‌کردن دکمه فقط روی رابط کاربری اثر دارد. سرور باید در هر عملیات حساس، هویت کاربر و مجوز دسترسی او را بررسی کند. APIها و Server Actions نیز باید مستقل از رابط کاربری محافظت شوند و نباید به مخفی‌بودن مسیر اعتماد کرد."},
        ],
    },

    # 120 — فرایند اصولی توسعه و بررسی کد
    {
        "category": "explanation", "language": "fa",
        "turns": [
            {"role": "user", "text": "برای اضافه‌کردن یک قابلیت Full-Stack به Next.js چه مراحلی را دنبال کنم؟"},
            {"role": "assistant", "text": "ابتدا رفتار مورد انتظار و ساختار فعلی پروژه را بررسی کن. سپس قرارداد داده و API را مشخص کن، ورودی‌ها را در سرور اعتبارسنجی کن، منطق و دسترسی دیتابیس را در لایه مناسب قرار بده و رابط کاربری را به آن متصل کن. حالت‌های loading، خطا، داده خالی و موفقیت را مدیریت کن. در پایان تغییرات را با تست‌های مرتبط بررسی کن و بدون اجرای واقعی، موفقیت تست یا ساخت را ادعا نکن."},
        ],
    },
     # 121 — ساخت Route Handler برای محصولات
    {
        "category": "code_generation", "language": "fa_code",
        "turns": [
            {"role": "user", "text": "در Next.js App Router یک API برای دریافت محصولات بساز."},
            {"role": "assistant", "text": "در فایل app/api/products/route.ts می‌توان تابع GET تعریف کرد. برای پروژه واقعی، داده‌ها باید از منبع معتبر مانند دیتابیس دریافت شوند.\n\nimport { NextResponse } from \"next/server\";\n\nexport async function GET() {\n  const products = [\n    { id: 1, name: \"لپ‌تاپ\", price: 25000000 },\n    { id: 2, name: \"ماوس\", price: 500000 },\n  ];\n\n  return NextResponse.json({ data: products });\n}"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Standalone code examples (Next.js / TypeScript) — to be filled in manually
# ---------------------------------------------------------------------------

# Each entry is a raw string of code (used for non-conversation training signal)

CODE_EXAMPLES: list[str] = [
    # Example template (do NOT remove — replace/add actual data here):
    # "// app/page.tsx\nexport default function HomePage() {\n  return <h1>سلام</h1>\n}\n",
]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_conversation(conv: dict) -> str:
    """Convert a conversation dict into the special-token format string."""
    parts: list[str] = []
    for turn in conv.get("turns", []):
        role = turn.get("role", "user")
        text = turn.get("text", "")
        marker = USER_TOKEN if role == "user" else ASSISTANT_TOKEN
        parts.append(f"{marker}{text}{END_TOKEN}")
    return "".join(parts)


def build_corpus() -> list[CodeDocument]:
    """Build the training corpus as CodeDocument objects.

    Conversations are serialized with special tokens.
    Standalone code examples are added as-is.

    Returns an empty list if no data has been added yet.
    """
    docs: list[CodeDocument] = []

    for conv in CONVERSATIONS:
        text = serialize_conversation(conv)
        if not text.strip():
            continue
        docs.append(CodeDocument(
            text=text,
            language=conv.get("language", "fa"),
            category=conv.get("category", "conversation"),
            source="original",
        ))

    for code in CODE_EXAMPLES:
        if not code.strip():
            continue
        docs.append(CodeDocument(
            text=code,
            language="nextjs",
            category="code",
            source="original",
        ))

    return docs


def get_special_tokens() -> dict[str, int]:
    """Return the special token strings that must be registered in the tokenizer.

    Actual IDs are assigned dynamically after BPE training by the training script.
    """
    return {
        USER_TOKEN: -1,
        ASSISTANT_TOKEN: -1,
        END_TOKEN: -1,
    }
