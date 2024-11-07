#include <iostream>
#include <stdlib.h>
#include "NN.hpp"


using namespace std;

int main()
{

    int lengthOfModel = 5;
    int modelFormat[] = {4,4,8,4,2};
    int LearningRate = 0.5;
    NN model(&modelFormat[0],lengthOfModel, LearningRate);
    double input[2] = {5,10};
    int numEpochs = 100;
    for (int i = 0; i < numEpochs; i++) {
        if (i == 0) {
            model.Model(&input[0]);
        }
        else {
            if (i % 5 == 0) {
                model.UpdateWeightsAndBiases();
            }
            model.Model(model.Output());
        }
    }
    cout << "output:" << endl;
    cout << model.OutputEpoch() << endl;
    double* arr = model.Output();
    for (int i = 0; i < 2; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
    model.DeleteNN();
    cout << "model deleted" << endl;
}
