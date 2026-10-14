// Author: Leonardo Gusson, 2157199

#ifndef CV_FINALPROJECT_CHESSBOARD_H
#define CV_FINALPROJECT_CHESSBOARD_H

#include <opencv2/opencv.hpp>

/**
 * @enum ChessPieceCategory
 * @brief Enumeration representing the different categories of chess pieces.
 */
enum ChessPieceCategory {
    BISHOP = 0,
    KING = 1,
    KNIGHT = 2,
    PAWN = 3,
    QUEEN = 4,
    ROOK = 5,
    NONE = -1,
};

/**
 * @class ChessBoard
 * @brief Class responsible for rendering a chessboard with its pieces.
 */
class ChessBoard {
public:
    /**
     * @brief Default constructor. Loads and pre-processes chess piece icons.
     */
    ChessBoard();

    /**
     * @brief Draws the complete chessboard with the provided pieces.
     * @param pieces Vector of 64 elements representing the board state.
     * @param isFirstWhite True if the top-left square is white.
     * @return A cv::Mat containing the rendered chessboard image.
     */
    cv::Mat drawChessboard(std::vector<std::pair<int, ChessPieceCategory> > &pieces, bool isFirstWhite);

private:
    /**
     * @brief Retrieves the correct pre-processed piece image.
     * @param category The type of the chess piece.
     * @param isWhiteBackground True if the piece is on a white square.
     * @return The specific piece image.
     */
    cv::Mat getPieceImg(ChessPieceCategory category, bool isWhiteBackground);

    /**
     * @brief Replaces the alpha channel of a PNG icon with a solid background.
     * @param inputImage The raw 4-channel image.
     * @param whiteBackground True for white background, false for black.
     * @return A 3-channel image with the applied background.
     */
    cv::Mat processIcon(const cv::Mat &inputImage, bool whiteBackground);

    // Pre-processed image for pieces on different backgrounds
    cv::Mat pawnOnBlack;
    cv::Mat pawnOnWhite;
    cv::Mat knightOnBlack;
    cv::Mat knightOnWhite;
    cv::Mat bishopOnBlack;
    cv::Mat bishopOnWhite;
    cv::Mat rookOnBlack;
    cv::Mat rookOnWhite;
    cv::Mat queenOnBlack;
    cv::Mat queenOnWhite;
    cv::Mat kingOnBlack;
    cv::Mat kingOnWhite;
    bool loadingError = false;

    // Chess piece images paths
    const std::string pawnPath = INPUT_FILES_DIR + std::string("pawn.png");
    const std::string knightPath = INPUT_FILES_DIR + std::string("knight.png");
    const std::string bishopPath = INPUT_FILES_DIR + std::string("bishop.png");
    const std::string rookPath = INPUT_FILES_DIR + std::string("rook.png");
    const std::string kingPath = INPUT_FILES_DIR + std::string("king.png");
    const std::string queenPath = INPUT_FILES_DIR + std::string("queen.png");
};


#endif //CV_FINALPROJECT_CHESSBOARD_H
