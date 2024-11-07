#include <stdlib.h>
#include <random>
#include <ctime>
#include <iostream>
#include "Layer.hpp"

using namespace std;

void initializeRandomWeightsAndBiases(double** weightsArray, double* biasesArray, double* nodeArray, int n_nodes, int n_inputs);

Layer::Layer() : n_inputs(0), n_nodes(0), weightsArray(nullptr), biasesArray(nullptr), nodeArray(nullptr) {}

Layer::Layer(int n_input, int n_node) : n_inputs(n_input), n_nodes(n_node) {
    weightsArray = new double* [n_nodes];
    for (int i = 0; i < n_nodes; i++) {
        weightsArray[i] = new double[n_inputs];
    }

    biasesArray = new double[n_nodes];
    nodeArray = new double[n_nodes];

    initializeRandomWeightsAndBiases(&weightsArray[0],&biasesArray[0],&nodeArray[0],n_nodes,n_inputs);
}

void Layer::forward(double inputsArray[]) {
    for (int i = 0; i < n_nodes; i++) {
        for(int j = 0; j < n_inputs; j++) {
            nodeArray[i] += weightsArray[i][j] * inputsArray[j];
        }
        nodeArray[i] += biasesArray[i];
    }
}

void Layer::activation() {
    for (int i = 0; i < n_nodes; i++) {
        if(nodeArray[i] < 0) {
            nodeArray[i] = 0;
        }
    }
}

double* Layer::getNodeArray() {
    return &nodeArray[0];
}

void Layer::DeleteLayer() {
    delete[] weightsArray;
    delete[] biasesArray;
    delete[] nodeArray;
}

void Layer::UpdateWeights(int learningRate) {
    for (int i = 0; i < n_nodes; i++) {
        for(int j = 0; j < n_inputs; j++) {
            weightsArray[i][j] = weightsArray[i][j] * learningRate;
        }
    }
}

void Layer::UpdateBiases(int learningRate) {
    for (int i = 0; i < n_nodes; i++) {
        biasesArray[i] = biasesArray[i] * learningRate;
    }
}

void initializeRandomWeightsAndBiases(double** weightsArray, double* biasesArray, double* nodeArray, int n_nodes, int n_inputs) {
    default_random_engine generator(time(0));
    normal_distribution distribution(0.0, 0.01);

    for (int i = 0; i < n_nodes; i++) {
        nodeArray[i] = 0;
        for (int j = 0; j < n_inputs; j++) {
            weightsArray[i][j] = distribution(generator) * sqrt(2.0 / n_inputs);
        }
    }

    for (int i = 0; i < n_nodes; i++) {
        biasesArray[i] = 0.1;
    }
}

