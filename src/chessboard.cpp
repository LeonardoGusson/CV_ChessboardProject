// Author: Leonardo Gusson 2157199

#include "chessboard.h"

ChessBoard::ChessBoard() {
    cv::Mat pawn = cv::imread(pawnPath, cv::IMREAD_UNCHANGED);
    if (pawn.empty() || pawn.channels() != 4) loadingError = true;
    cv::Mat knight = cv::imread(knightPath, cv::IMREAD_UNCHANGED);
    if (knight.empty() || knight.channels() != 4) loadingError = true;
    cv::Mat bishop = cv::imread(bishopPath, cv::IMREAD_UNCHANGED);
    if (bishop.empty() || bishop.channels() != 4) loadingError = true;
    cv::Mat rook = cv::imread(rookPath, cv::IMREAD_UNCHANGED);
    if (rook.empty() || rook.channels() != 4) loadingError = true;
    cv::Mat king = cv::imread(kingPath, cv::IMREAD_UNCHANGED);
    if (king.empty() || king.channels() != 4) loadingError = true;
    cv::Mat queen = cv::imread(queenPath, cv::IMREAD_UNCHANGED);
    if (queen.empty() || queen.channels() != 4) loadingError = true;

    if (!loadingError) {
        pawnOnWhite = processIcon(pawn, true);
        pawnOnBlack = processIcon(pawn, false);
        knightOnWhite = processIcon(knight, true);
        knightOnBlack = processIcon(knight, false);
        rookOnWhite = processIcon(rook, true);
        rookOnBlack = processIcon(rook, false);
        bishopOnWhite = processIcon(bishop, true);
        bishopOnBlack = processIcon(bishop, false);
        queenOnWhite = processIcon(queen, true);
        queenOnBlack = processIcon(queen, false);
        kingOnWhite = processIcon(king, true);
        kingOnBlack = processIcon(king, false);
    }
}

cv::Mat ChessBoard::drawChessboard(std::vector<std::pair<int, ChessPieceCategory> > &pieces, bool isFirstWhite) {
    if (loadingError || pieces.size() != 64) return cv::Mat();

    constexpr int imgSize = 416;
    constexpr int cellSize = 45;
    constexpr int lineThickness = 1;
    constexpr int outerPadding = 27;
    constexpr int piecePadding = 5; // Assuming chess piece icon is 35x35

    cv::Mat img(imgSize, imgSize, CV_8UC3, cv::Scalar(255, 255, 255));
    constexpr int boardWidth = (cellSize * 8) + (lineThickness * 2);

    // Draw a square around the chessboard
    const cv::Rect boundingBox(outerPadding, outerPadding, boardWidth, boardWidth);
    cv::rectangle(img, boundingBox, cv::Scalar(0, 0, 0), lineThickness);

    // Draw chessboard and pieces
    constexpr int startX = outerPadding + lineThickness;
    constexpr int startY = outerPadding + lineThickness;
    int cellCounter = 0;
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            // current cell color
            bool isCellWhite;
            if (isFirstWhite) {
                isCellWhite = ((row + col) % 2 == 0);
            } else {
                isCellWhite = ((row + col) % 2 != 0);
            }
            cv::Scalar cellColor = isCellWhite ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);

            // current cell position
            const int cellStartX = startX + col * cellSize;
            const int cellStartY = startY + row * cellSize;
            const cv::Rect cellRect(cellStartX, cellStartY, cellSize, cellSize);
            cv::rectangle(img, cellRect, cellColor, cv::FILLED);

            // draw the chess piece
            if (pieces[cellCounter].second != ChessPieceCategory::NONE) {
                cv::Mat piece = getPieceImg(pieces[cellCounter].second, isCellWhite);
                cv::Rect roi(cellStartX + piecePadding, cellStartY + piecePadding, piece.cols, piece.rows);
                piece.copyTo(img(roi));
            }

            cellCounter++;
        }
    }
    return img;
}

cv::Mat ChessBoard::getPieceImg(const ChessPieceCategory category, const bool isWhiteBackground) {
    if (category == ChessPieceCategory::PAWN)
        return isWhiteBackground ? pawnOnWhite : pawnOnBlack;
    if (category == ChessPieceCategory::KNIGHT)
        return isWhiteBackground ? knightOnWhite : knightOnBlack;
    if (category == ChessPieceCategory::BISHOP)
        return isWhiteBackground ? bishopOnWhite : bishopOnBlack;
    if (category == ChessPieceCategory::ROOK)
        return isWhiteBackground ? rookOnWhite : rookOnBlack;
    if (category == ChessPieceCategory::KING)
        return isWhiteBackground ? kingOnWhite : kingOnBlack;
    if (category == ChessPieceCategory::QUEEN)
        return isWhiteBackground ? queenOnWhite : queenOnBlack;
    return cv::Mat();
}

cv::Mat ChessBoard::processIcon(const cv::Mat &inputImage, bool whiteBackground) {
    cv::Mat resizedImg;
    cv::resize(inputImage, resizedImg, cv::Size(35, 35), 0, 0, cv::INTER_AREA);

    std::vector<cv::Mat> channels;
    cv::split(resizedImg, channels);

    const cv::Mat contentMask = channels[3] > 0;
    const cv::Mat backgroundMask = channels[3] == 0;

    // Draw the chess piece
    channels[0].setTo(255, contentMask);
    channels[1].setTo(0, contentMask);
    channels[2].setTo(0, contentMask);

    // Draw the background
    if (whiteBackground) {
        channels[0].setTo(255, backgroundMask);
        channels[1].setTo(255, backgroundMask);
        channels[2].setTo(255, backgroundMask);
    } else {
        channels[0].setTo(0, backgroundMask);
        channels[1].setTo(0, backgroundMask);
        channels[2].setTo(0, backgroundMask);
    }

    cv::Mat finalImage;
    channels.pop_back();
    cv::merge(channels, finalImage);

    return finalImage;
}

