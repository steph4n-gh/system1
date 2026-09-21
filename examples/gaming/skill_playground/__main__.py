import argparse
from pathlib import Path
from .experiment import experiment


def main():
    parser = argparse.ArgumentParser(description="Teach and evaluate a local network of System1 skills")
    parser.add_argument("--output-dir", type=Path, default=Path(".system1/skill-playground"))
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--serve", action="store_true", help="Open the local interactive playground")
    parser.add_argument("--port", type=int, default=8789)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 200:
        parser.error("episodes must be between 1 and 200")
    if not args.serve or not (args.output_dir / "report.json").exists():
        experiment(args.output_dir, args.episodes, args.seed)
    if args.serve:
        from .server import serve
        if not 1024 <= args.port <= 65535:
            parser.error("port must be between 1024 and 65535")
        serve(args.output_dir, args.port)


if __name__ == "__main__":
    main()
