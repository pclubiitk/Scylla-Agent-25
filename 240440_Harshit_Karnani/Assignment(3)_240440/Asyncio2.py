import aiohttp
import aiofiles
import asyncio
import time
import os

os.makedirs("./async", exist_ok=True)


async def write_genre(file_name):

    async with aiohttp.ClientSession() as session:
        async with session.get("https://binaryjazz.us/wp-json/genrenator/v1/genre/") as response:
            genre = await response.json()

    async with aiofiles.open(file_name, "w") as new_file:
        print(f'Writing "{genre}" to "{file_name}"...')
        await new_file.write(str(genre))

# To call an async function, you must either use the await keyword from another async function or loop = asyncio.get_event_loop()


async def main():
    tasks = []

    for i in range(5):
        tasks.append(write_genre(f"./async/new_file{i}.txt"))

    await asyncio.gather(*tasks)
    await asyncio.sleep(10)

start_time = time.time()
asyncio.run(main())
end_time = time.time()
print(f"Time taken with asyncio: {end_time - start_time} seconds")
