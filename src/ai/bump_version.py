import argparse
import re
import os

VERSION_FILE = os.path.join(os.path.dirname(__file__), "app", "version.py")

def bump_version(part):
    with open(VERSION_FILE, "r") as f:
        content = f.read()

    match = re.search(r'__version__ = "(\d+)\.(\d+)\.(\d+)"', content)
    if not match:
        print(f"Error: Could not find version string in {VERSION_FILE}")
        return

    major, minor, patch = map(int, match.groups())

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        print("Invalid part. Use major, minor, or patch.")
        return

    new_version = f"{major}.{minor}.{patch}"
    new_content = re.sub(r'__version__ = ".*"', f'__version__ = "{new_version}"', content)

    with open(VERSION_FILE, "w") as f:
        f.write(new_content)

    print(f"Bumps version to {new_version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bump version number.")
    parser.add_argument("part", choices=["major", "minor", "patch"], help="Part of version to bump")
    args = parser.parse_args()
    bump_version(args.part)
