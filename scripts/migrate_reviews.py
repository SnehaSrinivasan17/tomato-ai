"""Explicit one-time/idempotent migration. Secrets and provider errors are never printed."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--secrets-file')
    parser.add_argument('--report',default='phase7_results/review_migration.json')
    args=parser.parse_args()
    if args.secrets_file:
        from config import load_secrets_for_script
        load_secrets_for_script(args.secrets_file)
    from src.review.migrate import migrate
    try:report=migrate()
    except Exception:
        print('Migration could not complete. Check private database settings, connectivity and schema permissions; existing records were not overwritten.')
        return 1
    path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
