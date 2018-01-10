using System;
using System.Collections.Generic;
using Layer = System.Collections.Generic.List<Neuron>;

public class NeuralNetwork : IComparable<NeuralNetwork>
{
    static float _recentAverageSmoothingFactor = 0.15f;
    List<Layer> _layers = new List<Layer>();
    float _error;
    float _recentAverageError;
    float _fitness;

    public NeuralNetwork(List<int> topology)
    {
        int numLayers = topology.Count;
        for (int layerNum = 0; layerNum < numLayers; ++layerNum)
        {
            var layer = new Layer();
            int numOutputs = layerNum == topology.Count - 1 ? 0 : topology[layerNum + 1];
            for (int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
            {
                layer.Add(new Neuron(numOutputs, neuronNum));
            }
            layer[layer.Count - 1].SetOutputValue(1.0f);
            _layers.Add(layer);
        }
    }

    public NeuralNetwork(NeuralNetwork other)
    {
        _layers.Clear();
        for (int layerNum = 0; layerNum < other._layers.Count; ++layerNum)
        {
            var layer = new Layer();
            int numOutputs = layerNum == other._layers.Count - 1 ? 0 : other._layers[layerNum + 1].Count;
            for (int neuronNum = 0; neuronNum < other._layers[layerNum].Count; ++neuronNum)
            {
                layer.Add(new Neuron(numOutputs, neuronNum));
            }
            layer[layer.Count - 1].SetOutputValue(0.2f);
            _layers.Add(layer);
        }
    }

    public void CloneValues(NeuralNetwork other)
    {
        for (int layerNum = 0; layerNum < other._layers.Count; ++layerNum)
        {
            var layer = _layers[layerNum];
            var otherLayer = other._layers[layerNum];
            for (int neuronNum = 0; neuronNum < other._layers[layerNum].Count; ++neuronNum)
            {
                layer[neuronNum].CloneValues(otherLayer[neuronNum]);
            }
        }
    }

    public void CrossOver(NeuralNetwork other)
    {
        for (int layerNum = 0; layerNum < other._layers.Count; ++layerNum)
        {
            var layer = _layers[layerNum];
            var otherLayer = other._layers[layerNum];
            for (int neuronNum = 0; neuronNum < other._layers[layerNum].Count; ++neuronNum)
            {
                layer[neuronNum].CrossOver(otherLayer[neuronNum]);
            }
        }
    }

    public void FeedForward(List<float> inputVal)
    {
        for (int i = 0; i < inputVal.Count; ++i)
        {
            _layers[0][i].SetOutputValue(inputVal[i]);
        }

        //forward propagate
        for (int layerNum = 1; layerNum < _layers.Count; ++layerNum)
        {
            Layer prevLayer = _layers[layerNum - 1];
            for (int n = 0; n < _layers[layerNum].Count - 1; ++n)
            {
                _layers[layerNum][n].FeedForward(prevLayer);
            }
        }
    }

    public List<float> GetResults()
    {
        var retVal = new List<float>();
        Layer outputLayer = _layers[_layers.Count - 1];
        for (int n = 0; n < outputLayer.Count - 1; ++n)
        {
            retVal.Add(outputLayer[n].GetOutputValue());
        }
        return retVal;
    }

    public void Mutate()
    {
        for (int layerNum = 1; layerNum < _layers.Count; ++layerNum)
        {
            Layer prevLayer = _layers[layerNum - 1];
            for (int n = 0; n < _layers[layerNum].Count - 1; ++n)
            {
                _layers[layerNum][n].Mutate();
            }
        }
    }

    public void BackPropagation(List<float> targetVals)
    {
        //Calculate overall net error (RMS of output neuron errors)
        Layer outputLayer = _layers[_layers.Count - 1];
        _error = 0.0f;
        for (int n = 0; n < outputLayer.Count - 1; ++n)
        {
            float delta = targetVals[n] - outputLayer[n].GetOutputValue();
            _error += delta * delta;
        }
        _error /= outputLayer.Count - 1; // get average error squared
        _error = (float)Math.Sqrt(_error); // calculate RMS

        //implement a recent average measurement
        _recentAverageError =
            (_recentAverageError * _recentAverageSmoothingFactor + _error)
            / (_recentAverageSmoothingFactor + 1.0f);
        //Calculate output layer gradients

        for (int n = 0; n < outputLayer.Count - 1; ++n)
        {
            outputLayer[n].CalcOutputGradients(targetVals[n]);
        }

        //Calculate gradients on hidden layers

        for (int layerNum = _layers.Count - 2; layerNum > 0; --layerNum)
        {
            Layer hiddenLayer = _layers[layerNum];
            Layer nextLayer = _layers[layerNum + 1];
            for (int n = 0; n < hiddenLayer.Count; ++n)
            {
                hiddenLayer[n].CalcHiddenGradients(nextLayer);
            }
        }

        //for all layers from outputs to first hidden layer
        //update connection weights
        for (int layerNum = _layers.Count - 1; layerNum > 0; --layerNum)
        {
            Layer layer = _layers[layerNum];
            Layer prevLayer = _layers[layerNum - 1];
            for (int n = 0; n < layer.Count - 1; ++n)
            {
                layer[n].UpdateInputWeights(prevLayer);
            }
        }
    }

    public void SetFitness(float newValue)
    {
        _fitness = newValue;
    }

    public void AddFitness(float delta)
    {
        _fitness += delta;
    }

    public float GetFitness()
    {
        return _fitness;
    }

    public int CompareTo(NeuralNetwork other)
    {
        if (other == null) return 1;

        if (_fitness < other._fitness)
            return 1;
        else if (_fitness > other._fitness)
            return -1;
        else
            return 0;

    }
}