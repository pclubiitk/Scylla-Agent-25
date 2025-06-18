import concurrent.futures
import time


if __name__ == "__main__":
    pow_list = [i for i in range(1000000, 1000016)]

    print("Starting...")
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(pow, i, i) for i in pow_list]

    for f in concurrent.futures.as_completed(futures):
        print("ok")

    end = time.time()
    print(f"Time taken to complete : {(end - start)}")
