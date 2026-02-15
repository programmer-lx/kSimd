import argparse
import subprocess
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent

def main():
    parser = argparse.ArgumentParser(description="Quick G++ Compiler Wrapper")
    
    # --file 参数
    parser.add_argument("--file", required=True, help="Path to the .cpp file")
    
    # --includes 参数 (支持多个，例如 --includes ./dir1 ./dir2)
    parser.add_argument("--includes", nargs='*', default=[], help="List of include directories")
    
    # --macros 参数 (支持多个，例如 --macros MY_MACRO=1 DEBUG)
    parser.add_argument("--macros", nargs='*', default=[], help="List of macros to define")

    args = parser.parse_args()

    # 1. 检查文件是否存在
    cpp_file = os.path.abspath(args.file)
    if not os.path.exists(cpp_file):
        print(f"Error: File '{cpp_file}' not found.")
        sys.exit(1)

    # 2. 构造目标文件路径 (.cpp -> .o)
    # 比如 C:/test/main.cpp 变成 C:/test/main.o
    base_path = os.path.splitext(cpp_file)[0]
    output_file = base_path + ".o"

    # 3. 构建 g++ 命令行
    # -c 表示只编译不链接
    command = [
        "g++",
        "-c",
        "-std=c++20",
        "-fno-exceptions",
        "-O2",
        "-fno-asynchronous-unwind-tables",
        f'-I{project_root / "kSimd"}',
        cpp_file,
        "-o",
        output_file
    ]

    # 添加包含路径
    for inc in args.includes:
        command.append(f"-I{inc}")

    # 添加宏定义
    for macro in args.macros:
        command.append(f"-D{macro}")

    # 打印最终执行的命令，方便调试
    print(f"Executing: {' '.join(command)}")

    # 4. 执行编译
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(script_dir))
        if result.returncode == 0:
            print(f"Successfully compiled: {output_file}")
            # 自动执行 nm 查看符号
            print("\n--- Symbols ---")
            subprocess.run(["nm", "-C", output_file])
        else:
            print("Compilation Failed:")
            print(result.stderr)
    except FileNotFoundError:
        print("Error: 'g++' or 'nm' not found in PATH.")

if __name__ == "__main__":
    main()