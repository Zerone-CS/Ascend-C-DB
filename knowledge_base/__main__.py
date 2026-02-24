"""Allow running the package directly: python -m knowledge_base [build|query] ..."""

import sys


def main():
    usage = (
        "Usage: python -m knowledge_base <command> [options]\n"
        "\n"
        "Commands:\n"
        "  build   \u6784\u5efa\u77e5\u8bc6\u5e93\u7d22\u5f15\n"
        "  query   \u67e5\u8be2\u77e5\u8bc6\u5e93\n"
        "\n"
        "Examples:\n"
        "  python -m knowledge_base build\n"
        "  python -m knowledge_base build --dense\n"
        "  python -m knowledge_base query -q 'Tiling\u5207\u5206'\n"
        "  python -m knowledge_base query              # \u4ea4\u4e92\u6a21\u5f0f\n"
    )

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(usage)
        sys.exit(0)

    command = sys.argv.pop(1)

    if command == "build":
        from knowledge_base.build_kb import main as build_main
        build_main()
    elif command == "query":
        from knowledge_base.query_kb import main as query_main
        query_main()
    else:
        print(f"Unknown command: {command}\n")
        print(usage)
        sys.exit(1)


if __name__ == "__main__":
    main()
