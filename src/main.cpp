#include <iostream>
#include "io_functions.h"
#include "chess_classifier.h"
#include "board_detector.h"
#include "evaluation.h"

int main() {
    /*******************************************************************************/
    /************************** Model training or loading **************************/
    /*******************************************************************************/
    bool training = askForBinaryOptions(
        std::string("Choose whether to train the model from scratch or to load a pre-trained model."),
        'Y',
        std::string("training the model"),
        'N',
        std::string("loading the pre-trained model.")
    );

    ChessClassifier classifier{};

    std::cout << "Dataset initialization" << std::endl;
    auto dataset1 = Dataset1();
    std::cout << ">>> Executed" << std::endl;

    if (!training) {
        std::cout << "Trying to load a pre-trained model" << std::endl;
        bool loadSuccess = classifier.loadTrainedModel();

        if (!loadSuccess) {
            std::cout << "Model loading failed. Training anyway." << std::endl;
            training = true;
        } else {
            std::cout << ">>> Executed" << std::endl;
        }
    }

    if (training) {
        std::cout << "Loading a further dataset (only used for training purposes)" << std::endl;
        auto dataset2 = Dataset2();
        std::cout << ">>> Executed" << std::endl;

        std::cout << "Training the model on " << dataset1.getAllTrainImagesPaths().size() + dataset2.
                getTrainingSetSize() << " images" << std::endl;
        classifier.train(dataset1, dataset2);
        if (!classifier.saveTrainedModel()) {
            std::cerr << "Warning: failed to persist trained classifier." << std::endl;
        }
        std::cout << ">>> Executed" << std::endl;
    }

    /*******************************************************************************/
    /************************************* Test ************************************/
    /*******************************************************************************/
    BoardDetector boardDetector{};

    // Testing mode choice
    const bool singleImgTest = askForBinaryOptions(
        std::string("Choose the test system."),
        'S',
        std::string("single image test"),
        'A',
        std::string("testing the whole dataset at once.")
    );

    // TEST & Evaluation Metrics
    if (singleImgTest) {
        int index = dataset1.askForTestImageIndex();
        Evaluation::runSingleImageEvaluation(dataset1, boardDetector, classifier, index, 0.30f);
    } else {
        const DatasetEvaluationSummary summary = Evaluation::runDatasetEvaluation(
            dataset1, boardDetector, classifier, 0.30f);
        Evaluation::printDatasetEvaluationSummary(summary);
        std::cout << std::endl;
    }


    std::cout << "Metrics computation completed." << std::endl;
}
