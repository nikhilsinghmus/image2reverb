import sys
import os
from google_images_download import google_images_download


def main():
    response = google_images_download.googleimagesdownload()
    for f in sys.argv[1:]:
        arguments = {
            "keywords": os.path.basename(f),
            "limit": 20,
            "print_urls": True
        }
        paths = response.download(arguments)
        print(paths)


if __name__ == "__main__":
    main()