import os, sys, time
import carla

hostname = sys.argv[1]

print("CARLA client:", carla.__file__)
start = time.time()
time.sleep(2)
while True:
    try:
        print("Waiting for CARLA")
        print(f"Trying to connect to {hostname} ({time.time() - start}s elapsed)")
        client = carla.Client(hostname, 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        break
    except:
        import traceback
        traceback.print_exc()
        time.sleep(2)