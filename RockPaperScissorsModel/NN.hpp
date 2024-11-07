//
// Created by Dylan Frajerman on 10/7/2024.
//
#include "Layer.hpp"
#ifndef NN_HPP
#define NN_HPP

class NN {
    Layer* LayerList;
    int epochCount;
    int len;
    int lenOfLayers;
    int LearningRate;
    int Epoch = 0;
    public:
    NN(int modelFormat[], int lenFormat, int learningRate);
    void UpdateWeightsAndBiases();
    void Model(double inputs[]);
    double* Output();
    void DeleteNN();
    int OutputEpoch();
};

#endif //NN_HPP
