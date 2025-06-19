import json
import threading
import time
from urllib.request import Request, urlopen


def write_genre(file_name):
    req = Request("https://binaryjazz.us/wp-json/genrenator/v1/genre/",
                  headers={"User-Agent": "Mozilla/5.0"})
    genre = json.load(urlopen(req))
    # print(genre)

    with open(file_name, "w") as new_file:
        print(f"Writing '{genre}' to '{file_name}'...")
        new_file.write(genre)


# write_genre("genre.txt")
start_time = time.time()

threads = []
for i in range(5):
    thread = threading.Thread(
        target=write_genre,
        args=[f"./threading/new_file{i}.txt"]
    )
    # You can pass into it the kwarg 'target' with a value of whatever function you would like to run on that thread
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

end_time = time.time()
print(f"Time taken with concurrency : {end_time - start_time} seconds")

# Same code without concurrency
start_time2 = time.time()

for i in range(5):
    write_genre(f"./threading/new_file{i+5}.txt")

end_time2 = time.time()
print(f"Time taken without concurrency : {end_time2 - start_time2} seconds")
