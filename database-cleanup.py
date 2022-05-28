# ----- this script will do a cleanup of all the content of the database -----
import os


def main():
    try:
        os.remove("images/summarize.txt")
    except Exception:
        print("There is no summarize !")

    try:
        list_of_files = os.listdir("images/test/")

        for folder in list_of_files:
            folder_content = os.listdir(f"images/test/{folder}/")
            for file in folder_content:
                os.remove(f"images/test/{folder}/{file}")

        list_of_files = os.listdir("images/train/")

        for folder in list_of_files:
            folder_content = os.listdir(f"images/train/{folder}/")
            for file in folder_content:
                os.remove(f"images/train/{folder}/{file}")

    except Exception:
        print("There are no file !")


if __name__ == "__main__":
    main()
