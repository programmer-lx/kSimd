import subprocess
import sys
from pathlib import Path
import platform


OS_NAME = platform.system().lower()

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
asm_root = project_root / "tests_asm"

compiler = ""
if OS_NAME == "windows":
    compiler = "g++"
elif OS_NAME == "linux":
    compiler = "g++-13"
else:
    raise RuntimeError(f"Unsupported platform: {OS_NAME}.")

def run_command(command):
    print(f"\n[RUNNING] {' '.join(command)}")
    subprocess.run(command, check=True)

def generate_asm_for_file(src_path: Path):
    """为单个 cpp 文件生成汇编"""
    asm_path = src_path.with_suffix(".s")

    command = [
        str(compiler),
        "-std=c++20",
        "-fno-exceptions",      # 禁止异常
        "-O2",
        "-fno-asynchronous-unwind-tables",
        "-S", str(src_path),
        "-o", str(asm_path),
        f'-I{project_root / "kSimd"}',
        f'-I{project_root / "tests_asm"}'
    ]

    run_command(command)
    print(f"[SUCCESS] Generated assembly: {asm_path}")

def main():
    if not project_root.is_dir():
        print(f"[ERROR] Root directory does not exist: {project_root}")
        sys.exit(1)

    # 遍历 asm_root 下所有 cpp 文件（递归）
    cpp_files = list(asm_root.rglob("*.cpp"))
    if not cpp_files:
        print(f"[WARNING] No .cpp files found in {asm_root}")
        return

    for cpp_file in cpp_files:
        try:
            generate_asm_for_file(cpp_file)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Compilation failed for {cpp_file} with code {e.returncode}")
        except Exception as e:
            print(f"[FATAL ERROR] {cpp_file}: {e}")

if __name__ == "__main__":
    main()
