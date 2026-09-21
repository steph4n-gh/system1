import argparse
from pathlib import Path
from .app import Session, serve


def main():
    parser=argparse.ArgumentParser(description='Teach a document-filing skill by demonstrating choices')
    parser.add_argument('--output-dir',type=Path,default=Path('.system1/teaching-by-doing'))
    parser.add_argument('--port',type=int,default=8791)
    parser.add_argument('--evaluate',action='store_true',help='Evaluate the explicit sample session in an empty output directory')
    args=parser.parse_args()
    if not 1024<=args.port<=65535: parser.error('Use a port from 1024 to 65535')
    if args.evaluate:
        session=Session(args.output_dir)
        if session.records: parser.error('Use a fresh output directory to preserve existing demonstrations')
        session.sample_session();session.teach()
        report=session.evaluate()
        print(f"New documents: {report['correct']}/{report['cases']} correct; {report['accepted_correct']}/{report['accepted']} accepted predictions correct")
        print(f"Unusual documents: {report['challenges_sent_to_review']}/{report['challenge_cases']} sent to review")
        print(f"Teaching: {report['teaching']['milliseconds']:.1f} ms; prediction median: {report['median_decision_ms']:.3f} ms; skill: {report['skill_bytes']} bytes")
        print(f"Full evidence: {args.output_dir/'report.json'}")
    else:
        serve(args.output_dir,args.port)


if __name__=='__main__': main()
