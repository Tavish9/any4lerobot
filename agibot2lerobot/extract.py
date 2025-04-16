import argparse
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract_tar(tar_path, target_dir):
    print(f"start extracting {tar_path}")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(target_dir)
        print(f"untar succeed: {tar_path} -> {target_dir}")
    except Exception as e:
        raise Exception(f"untar failed [{tar_path}]: {str(e)}")


def generate_tasks(source_root, dest_root):
    source_basename = source_root.stem
    for filepath in (source_root / "observations").glob("*/*.tar"):
        rel_dir = filepath.relative_to(source_root).parent

        dest_dir = dest_root / source_basename / rel_dir
        yield (filepath, dest_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path, help="src path")
    parser.add_argument("--dest", required=True, type=Path, help="dest path")
    parser.add_argument("--workers", type=int, default=24)

    args = parser.parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"src does not exist: {args.src}")

    os.makedirs(args.dest, exist_ok=True)

    tasks = generate_tasks(args.src, args.dest)

    total_tasks = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for src, dest in tasks:
            futures.append(executor.submit(extract_tar, src, dest))
            total_tasks += 1

        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
            except Exception as e:
                print("exception happens", e)
                with open("error_log.txt", "a") as f:
                    f.write(str(e) + "\n")
            print(f"Process: {i + 1}/{total_tasks}", end="\r")

    print("\nDone")


if __name__ == "__main__":
    main()
