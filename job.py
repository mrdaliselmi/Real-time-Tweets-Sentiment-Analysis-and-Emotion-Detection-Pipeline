import schedule
import time
import auto as auto
def job():
    auto.main()

# Schedule the job to run every day at a specific time (e.g., 8:00 AM)
schedule.every().day.at("08:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)