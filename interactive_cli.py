import subprocess
import time

# Kill any existing Streamlit processes
subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
time.sleep(2)

# Start Streamlit in background
process = subprocess.Popen(
    ["streamlit", "run", "notebook_03_indexing.py", "--server.port", "8505"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Give it time to start
time.sleep(5)

# Print the URL
print("Streamlit should be running at: http://localhost:8505")
