# ----- this script will do a cleanup of all the content of the database -----
import os


def main():
    try:
        os.remove("images/summarize.txt")
        list_of_files = os.listdir("images/")

        for folder in list_of_files:
            folder_content = os.listdir(f"images/{folder}/")
            for file in folder_content:
                os.remove(f"images/{folder}/{file}")
    except Exception:
        print("The files are already deleted !")


if __name__ == "__main__":
    main()
