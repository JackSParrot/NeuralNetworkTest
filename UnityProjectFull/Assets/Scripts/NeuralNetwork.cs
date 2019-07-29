using System;
using System.Collections.Generic;
using Layer = System.Collections.Generic.List<Neuron>;

public class NeuralNetwork : IComparable<NeuralNetwork>
{
    List<Layer> _layers = new List<Layer>();
    float _error;
    float _fitness;

    public NeuralNetwork(List<int> topology)
    {
        int numLayers = topology.Count;
        for (int layerNum = 0; layerNum < numLayers; ++layerNum)
        {
            var layer = new Layer();
            int numOutputs = layerNum == topology.Count - 1 ? 0 : topology[layerNum + 1];
            for (int neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum)
            {
                layer.Add(new Neuron(numOutputs, neuronNum));
            }
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

    public void FeedForward(List<float> inputVal)
    {
        for (int i = 0; i < inputVal.Count; ++i)
        {
            _layers[0][i].SetOutputValue(inputVal[i]);
        }

        for (int layerNum = 1; layerNum < _layers.Count; ++layerNum)
        {
            Layer prevLayer = _layers[layerNum - 1];
            foreach(var n in _layers[layerNum])
            {
                n.FeedForward(prevLayer);
            }
        }
    }

    public List<float> GetResults()
    {
        var retVal = new List<float>();
        Layer outputLayer = _layers[_layers.Count - 1];
        for (int n = 0; n < outputLayer.Count; ++n)
        {
            retVal.Add(outputLayer[n].GetOutputValue());
        }
        return retVal;
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

    public void Mutate()
    {
        for (int layerNum = 0; layerNum < _layers.Count; ++layerNum)
        {
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
        for (int n = 0; n < outputLayer.Count; ++n)
        {
            float delta = outputLayer[n].GetOutputDerivative() * (targetVals[n] - outputLayer[n].GetOutputValue());
            _error += delta * delta;
        }
        _error /= outputLayer.Count; // get average error squared
        _error = (float)Math.Sqrt(_error); // calculate RMS

        //Calculate output layer error
        for (int n = 0; n < outputLayer.Count; ++n)
        {
            outputLayer[n].CalcError(targetVals[n]);
        }

        //Calculate gradients on hidden layers
        for (int layerNum = _layers.Count - 2; layerNum > 0; --layerNum)
        {
            Layer hiddenLayer = _layers[layerNum];
            Layer nextLayer = _layers[layerNum + 1];
            for (int n = 0; n < hiddenLayer.Count; ++n)
            {
                hiddenLayer[n].CalcError(nextLayer);
            }
        }

        //for all layers from outputs to first hidden layer
        //update connection weights
        for (int layerNum = _layers.Count - 1; layerNum > 0; --layerNum)
        {
            Layer layer = _layers[layerNum];
            Layer prevLayer = _layers[layerNum - 1];
            for (int n = 0; n < layer.Count; ++n)
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

    public float GetError()
    {
        return _error;
    }

    public int CompareTo(NeuralNetwork other)
    {
        if (other == null) return -1;

        if (_fitness < other._fitness)
            return 1;
        else if (_fitness > other._fitness)
            return -1;
        else
            return 0;

    }
}