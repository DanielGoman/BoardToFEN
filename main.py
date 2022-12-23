from PIL import ImageGrab


def main():
    frame = ImageGrab.grab(bbox=(50, 50, 500, 500))
    frame.show()



if __name__ == "__main__":
    main()
