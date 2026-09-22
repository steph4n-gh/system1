import argparse
from pathlib import Path
from .app import Session, serve


def main():
    parser = argparse.ArgumentParser(description='Teach a document-filing skill by demonstrating choices')
    parser.add_argument('--output-dir', type=Path, default=Path('.system1/teaching-by-doing'))
    parser.add_argument('--port', type=int, default=8791)
    parser.add_argument('--evaluate', action='store_true', help='Review an explicit sample candidate in an empty output directory')
    parser.add_argument('--adopt', action='store_true', help='With --evaluate, explicitly adopt a passing sample candidate and preview it')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use a port from 1024 to 65535')
    if args.adopt and not args.evaluate:
        parser.error('--adopt requires --evaluate; use the page to adopt other candidates')
    if args.evaluate:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            parser.error('Use an empty output directory to preserve existing sessions and evidence')
        session = Session(args.output_dir)
        session.sample_session()
        report = session.teach()['candidate']
        candidate = report['candidate']
        print(f"Recurring development checks: {candidate['raw_correct']}/{candidate['count']} correct; {candidate['accepted_correct']}/{candidate['accepted']} accepted predictions correct")
        print(f"Candidate: {'passed' if report['passed'] else 'failed'}; approved skill unchanged")
        print(f"Candidate report: {session.lifecycle.report_path}")
        if args.adopt:
            if not report['passed']:
                parser.error('Candidate failed its checks and was not adopted')
            session.adopt()
            preview = session.evaluate()
            print(f"Adopted. Unusual documents: {preview['challenges_sent_to_review']}/{preview['challenge_cases']} sent to review")
            print(f"Prediction median: {preview['median_decision_ms']:.3f} ms; skill: {preview['skill_bytes']} bytes")
            print(f"Full preview: {args.output_dir/'preview-report.json'}")
        else:
            print('Open this output directory in the local page to review and adopt the candidate.')
    else:
        serve(args.output_dir, args.port)


if __name__ == '__main__':
    main()
