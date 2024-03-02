# BoardToFEN

# 📖 Introduction

This repository allows the user to screenshot window on his screen.

If this window contains a digital chessboard, the board will be detected and converted into a URL to the prominent chess 
sites [Chess.com](https://www.chess.com) and [Lichess](https://www.lichess.org). 
Additionally, one could simply copy the [FEN representation](https://www.chess.com/terms/fen-chess) of the given board.

---




# ▶️ Example

![](assets/Usage%20example.gif)

---




# 🚀 Motivation

The motivation of this project has arisen during the time when I studied and analyzed position from digital chess books.

In order to properly analyze the position I came across, sometimes I needed the assistance of a chess engine.

Unfortunately, it would take a while to copy the position to a chess analyzer, and hence this project 😊

---



# 🐥 Getting started


To clone the repository use
```bash
git clone https://github.com/DanielGoman/BoardToFEN.git
```

To boot the application interface use
```bash
cd BoardToFEN
pip install -r requirements.txt
python main.py
```

---




# 📁 Dataset

The dataset can be found in the following [drive](https://drive.google.com/file/d/1xc9vXlE55g4SCeJNspAnF_j-QJTNaoaZ/view?usp=drive_link)

The zip file contains 4 directories:
- **full_boards** - 34 images of different piece style boards in the starting position
- **replaced_king_queen** - 35 images of different piece style boards with only the queen and king on an opposite 
colored square compared to their starting position
- **squares** - `34 * 64 + 35 * 4` images of each square of those (each square of `full_boards` and only squares that 
contain pieces of `replaced_king_queen`)
- **labels** - json square-level labels files for each file



## Importing the dataset

### Linux
To import the dataset into the project directory use
```bash
wget -O "Board2FEN dataset.zip" -r --no-check-certificate 'https://drive.google.com/uc?export=download&id=1xc9vXlE55g4SCeJNspAnF_j-QJTNaoaZ' 
unzip  "Board2FEN dataset.zip" -d dataset/
```

### Windows
Just download from the link and unzip with your favorite unzipping software 😉



# 📜 Available scripts

 - [Board parser](src/data/board_parser.py) - parses all given directories of board image into a new `dataset/squares` 
directory of square images, as well as a respective `dataset/labels` labels file
 - [Train test split](src/data/train_test_split.py) - splits a prepared `dataset/labels.json` labels file into train and 
test jsons. Those jsons allow the train script to properly select the train and test image files.
 - [Train](src/model/train.py) - train a simple CNN for square classification - each item is a cropped image of a square 
that either contains a piece or not. The network learn to classify the piece type and color
 - [main](main.py) - runs the application that allows the user to select a window in their screen and (assuming it 
contains a board) convert it to FEN format and open it for analysis on their favorite chess site 😊



# Pipeline
This section will give a brief explanation of how the screenshot in converted into a FEn of the board that is in the image

## Identify board
First we identify the board in the image
1. We achieve this by applying a sharpening filter on the image, followed by
extracting the largest square shaped contour. 
2. This of course relies on the assumption that the largest square shaped contour is indeed the board. 
3. To me, it was reasonable to assume the user would likely take a screenshot of the board with some small margins.
4. Under this assumption, the chosen approach should usually work well.
5. After finding the location of the board in the image, we crop the image accordingly

## Split the board into squares
Now we split the board into squares
1. We achieve this by applying the Canny edge detector on the cropped image.
2. Following this we put our focus on the separating lines of the board (there should be 9 vertical and 9 horizontal such lines)
3. We find them by splitting the board into tentative 9 subregions twice (ones applying this vertically and once horizontally to find the vertical and horizontal lines respectively)
4. Then we select this line to be the longest edge in its subregion
5. Now that we have those 9 horizontal and 9 vertical lines, we are ready to split the board into a 8x8 grid of squares.

## Square classification
Now we reach the stage of classifying each square
1. We use a simple CNN with the following architecture
   1. 3 Convolution blocks
      1. 5x5 convolution
      2. Batch norm
      3. ReLU
      4. 2x2 Max pool with stride of 2
   2. 1 Linear layer followed by ReLU
   3. Final classifier Linear layer
2. The number of classes we use is 13
   1. Each square may contain either one of 6 white pieces, 6 black pieces, or be empty
   2. It might make more sense to have the model output 2 classifications - one for piece type and one for piece color.
   3. Although I started with this approach, I found it simpler to just work with single output with 13 classes.

## Conversion to FEN
All that remains is to convert the 8x8 predictions to a [FEN representation](https://www.chess.com/terms/fen-chess)

