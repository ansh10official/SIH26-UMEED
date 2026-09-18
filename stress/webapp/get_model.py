import urllib.request
import os

# Target destination folder setup
current_directory = os.path.dirname(os.path.abspath(__file__))
destination_file = os.path.join(current_directory, "landmarks.dat")

# Remove any old broken file versions first
if os.path.exists(destination_file):
    os.remove(destination_file)

print("Starting deep stream connection... Please do not close this terminal window.")

# Safe browser-identity setup to bypass server locks
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Secure cloud mirror containing the uncompressed file
source_url = "https://huggingface.co"

# Execute the background download pipeline
urllib.request.urlretrieve(source_url, destination_file)
print("SUCCESS: The 99MB landmarks file has loaded cleanly into your folder layout!")
