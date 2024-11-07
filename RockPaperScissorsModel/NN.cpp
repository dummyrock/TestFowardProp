//
// Created by Dylan Frajerman on 10/7/2024.
//
#include <stdlib.h>
#include <iostream>
#include "NN.hpp"
#include "Layer.hpp"

using namespace std;

NN::NN(int modelFormat[],int lenFormat, int learningRate) {
    LearningRate =  learningRate;
    LayerList = new Layer[lenFormat - 1];
    lenOfLayers = lenFormat-1;

    for (int i = 0; i < lenFormat-1; i++) {
        LayerList[i] = Layer(modelFormat[i], modelFormat[i + 1]);
    }
}

void NN::Model(double inputs[]) {
    for (int i = 0; i < lenOfLayers ; i++) {
        if (i == 0) {
            LayerList[i].forward(&inputs[0]);
            LayerList[i].activation();
        }
        else if (i == lenOfLayers) {
            LayerList[i].forward(LayerList[i-1].getNodeArray());
        }
        else {
            LayerList[i].forward(LayerList[i-1].getNodeArray());
            LayerList[i].activation();
        }
    }
    Epoch++;
}

double* NN::Output() {
    return LayerList[lenOfLayers - 1].getNodeArray();
}

int NN::OutputEpoch() {
    return Epoch;
}

void NN::UpdateWeightsAndBiases() {
    for (int i = 0; i < lenOfLayers ; i++) {
        LayerList[i].UpdateWeights(LearningRate);
        LayerList[i].UpdateBiases(LearningRate);
    }
}

void NN::DeleteNN() {
    for (int i = 0; i < lenOfLayers - 1; i++) {
        LayerList[i].DeleteLayer();
    }
    delete [] LayerList;
}

