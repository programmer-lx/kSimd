from pathlib import Path
import sys
import re

# 汇编文件所在根目录
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
asm_dir = project_root / "tests_asm"

if not asm_dir.is_dir():
    print(f"[ERROR] Directory not found: {asm_dir}")
    sys.exit(1)

# 允许出现的 call 符号子串（部分匹配即可）
ALLOWED_CALLS = [
    "ksimd_test_no_inline", # 自定义的非内联函数
    "dyn_func_index",       # 动态 dispatch index
    "sinf",                 # std::sin
]

# 遍历所有 .s 文件（递归）
asm_files = list(asm_dir.rglob("*.s"))
if not asm_files:
    print(f"[ERROR] No assembly files found in {asm_dir}")
    sys.exit(1)

failed_files = []

for asm_file in asm_files:
    with asm_file.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("call"):
                m = re.match(r"call\s+(\S+)", line)
                if m:
                    symbol = m.group(1)
                    # 只要 call 符号不包含任何允许子串，就算失败
                    if not any(allowed in symbol for allowed in ALLOWED_CALLS):
                        print(f"[ERROR] {asm_file}: Non-inlined call '{symbol}' found on line {lineno}")
                        failed_files.append(asm_file)
                        break

if failed_files:
    print(f"\n[FAILED] {len(failed_files)} file(s) have non-inlined calls.")
    sys.exit(1)
else:
    print(f"[SUCCESS] All {len(asm_files)} assembly files passed call inlining check.")
