import time
import os


print("inside test! check sleep")

print("launch other script")
os.system("python test2.py")
time.sleep(1)
print("continue doing stuff in test!")