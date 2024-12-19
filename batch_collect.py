import subprocess
from datetime import datetime
from dateutil.relativedelta import relativedelta
from time import sleep

start_date = datetime(2019, 1, 1)
end_date = datetime(2020, 1, 1)

current_date = start_date

while current_date < end_date:
    next_month = current_date + relativedelta(days=1)
    command = [
        "python3", 
        "collect_data.py", 
        f"--begin={current_date.strftime('%Y/%m/%d')}", 
        f"--end={next_month.strftime('%Y/%m/%d')}"
    ]
    
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)
    sleep(5)
    
    current_date = next_month