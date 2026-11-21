"""Write a CSV table and publish each completed experiment in its own folder."""
import csv
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def run_directory(parent):
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.run_', dir=parent) as temporary:
        staging = Path(temporary)
        yield staging
        destination = staging.with_name(staging.name[1:])
        staging.rename(destination)
    print(f'Results saved in {destination}')


def write_table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
