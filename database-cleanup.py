# ----- this script will do a cleanup of all the content of the database -----
import os


def erase_data():
    try:
        os.remove("images/summarize.txt")
    except Exception:
        print("There is no summarize !")

    try:
        list_of_files = os.listdir("images/test/")
        for folder in list_of_files:
            for file in os.listdir(f"images/test/{folder}"):
                os.remove(f"images/test/{folder}/{file}")

        list_of_files = os.listdir("images/train/")
        for folder in list_of_files:
            for file in os.listdir(f"images/train/{folder}"):
                os.remove(f"images/train/{folder}/{file}")

    except Exception:
        print("There are no files !")


if __name__ == "__main__":
    erase_data()
