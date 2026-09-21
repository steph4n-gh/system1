import argparse
from pathlib import Path
from .battle import experiment


def main():
    parser=argparse.ArgumentParser(description='Teach and compare two local Pokemon battle skills')
    parser.add_argument('--output-dir',type=Path,default=Path('.system1/pokemon-teaching'))
    parser.add_argument('--episodes',type=int,default=60)
    parser.add_argument('--seed',type=int,default=8000)
    parser.add_argument('--serve',action='store_true')
    parser.add_argument('--port',type=int,default=8790)
    parser.add_argument('--rom',type=Path,help='Optional English Red/Blue ROM for a real emulator check')
    parser.add_argument('--rom-suite',action='store_true',help='Run the fixed 18-fixture real-engine comparison')
    args=parser.parse_args()
    if not 1<=args.episodes<=200 or not 1024<=args.port<=65535 or not 0<=args.seed<=1000000:
        parser.error('Use 1–200 episodes, a port from 1024–65535, and a seed from 0–1000000')
    if args.rom_suite and args.rom is None:
        parser.error('--rom-suite requires --rom')
    if not (args.serve or args.rom_suite) or not (args.output_dir/'report.json').exists():
        experiment(args.output_dir,args.episodes,args.seed)
    if args.rom_suite:
        from .rom_experiment import experiment as rom_experiment
        rom_experiment(args.rom,args.output_dir)
    if args.serve:
        from .server import serve
        serve(args.output_dir,args.port,args.rom)


if __name__=='__main__':
    main()
