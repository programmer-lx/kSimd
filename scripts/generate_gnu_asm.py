import subprocess
import sys
from pathlib import Path
import argparse
import platform

def run_command(command):
    print(f"\n[RUNNING] {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Generate GNU assembly from a C++ source file.")
    parser.add_argument("source_file", type=str, help="Path to the C++ source file")
    parser.add_argument(
        "--compiler", type=str, default="g++",
        help="C++ compiler to use (default: g++)"
    )
    parser.add_argument(
        "--flags", type=str, default="-O2 -fno-asynchronous-unwind-tables",
        help="Additional compiler flags"
    )
    args = parser.parse_args()

    # resolve 输入路径
    src_path = Path(args.source_file).resolve()
    if not src_path.is_file():
        print(f"[ERROR] Source file does not exist: {src_path}")
        sys.exit(1)

    # 输出汇编文件路径，同目录，扩展名 .s
    asm_path = src_path.with_suffix(".s")

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    include_dir = project_root / "include"

    # 构建 g++ 命令
    command = [
        args.compiler,
        "-std=c++20",
        "-fno-exceptions",      # 禁止异常
        "-S", str(src_path),
        "-o", str(asm_path),
        f"-I{include_dir}",
        f"-I{src_path.parent}"
    ]

    # 添加用户自定义编译选项
    if args.flags:
        command.extend(args.flags.split())

    try:
        run_command(command)
        print(f"\n[SUCCESS] Generated assembly file: {asm_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Compilation failed with return code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
