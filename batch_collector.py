import subprocess
from datetime import datetime
from dateutil.relativedelta import relativedelta
from time import sleep
import argparse

parser = argparse.ArgumentParser(description='Batch collector tool')
parser.add_argument('--begin', type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=True, help='Begin date in YYYY/MM/DD format')
parser.add_argument('--end',   type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=True, help='End date in YYYY/MM/DD format')
parser.add_argument('--script', required=True, help='python script to run')
args = parser.parse_args()

start_date = args.begin
end_date = args.end
script = args.script

current_date = start_date

while current_date < end_date:
    next_day = current_date + relativedelta(days=1)
    command = [
        'python3', 
        script,
        f'--begin={current_date.strftime("%Y/%m/%d")}', 
        f'--end={next_day.strftime("%Y/%m/%d")}'
    ]
    
    print(f'Running: {" ".join(command)}')
    subprocess.run(command, check=True)
    
    current_date = next_day