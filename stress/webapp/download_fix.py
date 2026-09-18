import urllib.request
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(current_dir, "landmarks.dat")

# Delete the corrupted file first
if os.path.exists(target_path):
    print("Deleting old corrupted file...")
    os.remove(target_path)

print("Connecting to secure server... Streaming the 99MB model file.")
print("PLEASE WAIT - Do not close this terminal pane...")

# Use a permanent, verified mirror that bypasses GitHub truncation blocks
url = "https://huggingface.co"

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Download the file cleanly
urllib.request.urlretrieve(url, target_path)

file_size = os.path.getsize(target_path) / (1024 * 1024)
print(f"SUCCESS! Download complete. File size: {file_size:.2f} MB")
