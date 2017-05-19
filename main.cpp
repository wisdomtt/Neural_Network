#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "cassert"
#include <stdlib.h>
using namespace std;

struct Connection
{
    double weight;
    unsigned deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;//POWERFUL declaration

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(Layer &prevLayer);
    void setOutputVal(double val);
    double getOutputVal(void);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double transferFunctionDerivative(double x);
    static double transferFunction(double x);
    static double randomWeight(void);
    double sumDOW(const Layer &nextLayer)const;
    double m_outputVal;
    vector<Connection> m_outputWeights; //weigh value for outputs
    unsigned m_myIndex;
    double m_gradient;
    static double eta; // [0.0, 1.0] overall net training weight
    static double alpha;// [0.0, n] Multiplier of last weight change (momentum)
};


class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals);
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum], the entire neural net
    double m_error;
    double m_recentAverageSmoothingFactor;
    double m_recentAverageError;
};


/////////////////////////////////Neural_Network/////////////////////////////////////
Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++)
    {
        m_layers.push_back(Layer());
        //Appends objects of Layers to the vector m_layers

        unsigned numOutputs =
                layerNum == topology.size() -1 ? 0 : topology[layerNum +1];
        //layerNum+1 equals the num of layers in the next layer

        //Now we must append neurons to said layer
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)//less than equal to to incorporate bias neuron
        {
            //numOutputs is the number of neurons in the next layer, neuronNum = myIndex
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));//most recent element has neurons appended onto it contained
        }
    }
}


void Net::feedForward(const vector <double> &inputVals)
{
    //Exception Handler, inputs equal neurons in first layer
    assert(inputVals.size() == m_layers[0].size()-1);

    //Assign (latch) the input values into the input neurons
    for(unsigned i = 0; i < inputVals.size(); ++i)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    //loop through each layer, and each neuron then feed forward

    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum -1];
        for(unsigned n = 0; n<m_layers[layerNum].size() - 1; ++n)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}


void Net::backProp(const vector<double> &targetVals)
{
    //Calculate overall net error (RMS OF OUTPUT NEURON ERRORS)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    //Solve for error per neuron in prev layer
    for(unsigned n = 0; n< outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n]*outputLayer[n].getOutputVal();
        m_error += delta*delta;
    }

    m_error /=outputLayer.size() - 1;//Get average error squared
    m_error = sqrt(m_error);//RMS

    //Implement a recent average measure
    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor+1.0);

    //Calculate output layer gradients
    for (unsigned n = 0; n<outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    //Calculate gradients on hidden layers
    for(unsigned layerNum = m_layers.size()-2; layerNum > 0; --layerNum)
    {
        Layer &currentHiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum +1];

        //calcHiddenGradients for every neuron in the current hidden layer
        for(unsigned n = 0; n<currentHiddenLayer.size(); ++n)
        {
            currentHiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    //For all layers from outputs to first hidden layer,
    //update connection weights.
    for(unsigned layerNum = m_layers.size() -1; layerNum>0; --layerNum)
    {
        Layer &currentLayer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        //updatesInputWeights for all neuron's in current layer
        for (unsigned n = 0; n < currentLayer.size() - 1; ++n)
        {
            currentLayer[n].updateInputWeights(prevLayer);
        }
    }

}


void Net::getResults(vector<double> &resultVals)
{
    resultVals.clear();
    //cout<<"m_layers.back().size() "<<m_layers.back().size()<<endl;
    for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
    {
        //Binds outputValues of all the neurons into the resultVals vector
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


/////////////////////////////////Neuron/////////////////////////////////////
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    //numOutputs is the number of neurons in the next layer excluding bias neuron
    //I don't have an explanation for myIndex atm tbh

    //Appends random weights upon initialization of the neuron
    for (unsigned c=0; c<numOutputs; c++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}
double Neuron::randomWeight()
{
    return rand()/double(RAND_MAX);
}
void Neuron::setOutputVal(double val)
{
    m_outputVal = val;
}
double Neuron::getOutputVal(void)
{
    return m_outputVal;
}
void Neuron::feedForward(Layer &prevLayer)
{
    double sum = 0.0;
    //Sum the previous layer's outputs which are our inputs
    //include the bias node from the previous layer, and multiply them by weights
    for(unsigned n = 0; n< prevLayer.size(); n++)
    {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    //Real life application of a sigmoid curve, WOW math
    m_outputVal = Neuron::transferFunction(sum);
}


void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}


double Neuron::sumDOW(const Layer &nextLayer)const
{
    double sum = 0.0;
    //Sum our contributions of the errors at the nodes we feed
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}


void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}


double Neuron::transferFunction(double x)
{
    //tanh - output range [-1.0, 1.0]
    return tanh(x);
}


double Neuron::transferFunctionDerivative(double x)
{
    //PS, tanh is a hyperbolic function, the more you know, and
    // sech^(x) is the derivative of tanh(x)
    return (1/cosh(x))*(1/cosh(x));
}


double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;


void Neuron::updateInputWeights(Layer &prevLayer)
{
    //The weights to be updated are in the COnections container
    //In the neurons in the preceding layer
    for (unsigned n = 0; n<prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight =
                //Individual input, magnified by the gradient and train rate;
                eta //overall net learning rate
                * neuron.getOutputVal()
                * m_gradient
                // also add momentum = a function of the previous delta weight
                + alpha //momentum
                  * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

    }
}


int main() {
    //Temporary debug fix for random weight. Must run this once for randomization to begin
    std::cout << rand() / double(RAND_MAX) << std::endl;
    vector<unsigned> topology;
    topology.push_back(2);//Input Layer has 2 core neurons

    //Deep layer encapsulation
    topology.push_back(4);
    topology.push_back(6);
    //Deep Layer Encapsulation

    topology.push_back(1);//Output Layer with one neuron
    Net *net = new Net(topology);
    vector<double> inputVals(2);
    vector<double> resultVals(1);
    vector<double> targetVals(1);

    //Training Process for an Or gate example
    for(int x = 0; x < 2000; x++)
    {
        //Random Binomial Input Values of either 0 or 1
        inputVals[0] = rand()%2;
        inputVals[1] = rand()%2;
        cout<<"First Val: "<<inputVals[0]<<endl;
        cout<<"Second Val: "<<inputVals[1]<<endl;
        net->feedForward(inputVals);

        //Target Value
        targetVals[0] = ((inputVals[0] == 1)|(inputVals[1] == 1) ? 1 : 0);
        cout<<"Target Val: "<<targetVals[0]<<endl;

        net->backProp(targetVals);//Adjusts Weights
        net->getResults(resultVals);
        //Print out NN's results
        for(int x=0;x<resultVals.size(); x++)
        {
            cout<<"Result " << x << ": "<<resultVals[x]<<endl<<endl;
        }
    }
    //Loops through UI
    while(true)
    {
        vector<string> s_UserInput(2);
        vector<double> UserInput(2);

        //First Value
        cout<< "(Enter x to escape) First Val: "<<endl;
        getline(cin, s_UserInput[0]);//Record line
        assert(s_UserInput[0]!="x");//Escape
        UserInput[0] = atof(s_UserInput[0].c_str());//Convert String to double

        //Second Value
        cout<<"(Enter x to escape) Second Val: "<<endl;
        getline(cin, s_UserInput[1]);
        assert(s_UserInput[1]!="x");
        UserInput[1] = atof(s_UserInput[1].c_str());

        net->feedForward(UserInput);//Send Input Values into the Neural Net
        net->getResults(resultVals);//Retrieve results
        cout<<resultVals[0]<<endl;//Print out results
    }
    return 0;
}
