#ifndef LAYER_HPP
#define LAYER_HPP

class Layer {
    double ** weightsArray;
    double* biasesArray;
    double* nodeArray;
    int n_nodes;
    int n_inputs;
public:
    Layer();
    Layer(int n_inputs, int n_nodes);
    void forward(double inputsArray[]);
    void activation();
    double* getNodeArray();
    void DeleteLayer();
    void UpdateWeights(int learningRate);
    void UpdateBiases(int learningRate);
};

#endif //LAYER_HPP