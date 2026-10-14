"""Download the published human lineup data (Li & Cook, `visage` R package).

License is not stated in the repository: get the authors' permission before publishing results.
"""
import pathlib, urllib.request

BASE = "https://raw.githubusercontent.com/TengMCing/visage/master"
FILES = {
    "vi_survey.rda": f"{BASE}/data/vi_survey.rda",      # 8,860 human evaluations
    "vi_lineup.rds": f"{BASE}/data-raw/vi_lineup.rds",  # plot data for 1,176 lineups (64 MB)
}
DATA = pathlib.Path(__file__).parent / "data"

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = DATA / name
        if dest.exists():
            print("exists:", dest)
            continue
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    print("done")
