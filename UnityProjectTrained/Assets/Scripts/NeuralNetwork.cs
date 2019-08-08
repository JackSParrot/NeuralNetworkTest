using System.Collections.Generic;
using Layer = System.Collections.Generic.List<Neuron>;

public class NeuralNetwork
{
    List<Layer> _layers = new List<Layer>();
    float _fitness = 0f;
    List<float> _output = new List<float>();
    List<float> _errors = new List<float>();

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
            var bias = new Neuron(numOutputs, topology[layerNum]);
            bias.SetOutputValue(1f);
            layer.Add(bias);
            _layers.Add(layer);
        }
        _fitness = 0f;
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
        _fitness = other._fitness;
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
            for(int i = 0; i < _layers[layerNum].Count - 1; ++i)
            {
                var n = _layers[layerNum][i];
                n.FeedForward(prevLayer);
            }
        }
        _output.Clear();
        Layer outputLayer = _layers[_layers.Count - 1];
        for (int n = 0; n < outputLayer.Count - 1; ++n)
        {
            _output.Add(outputLayer[n].GetOutputValue());
        }
    }

    public void Train(List<float> inputs, List<float> desiredOutput)
    {
        FeedForward(inputs);
        CalculateError(desiredOutput);
        BackPropagate();
    }

    void CalculateError(List<float> desiredOutput)
    {
        var error = 0f;
        for (int i = 0; i < desiredOutput.Count; ++i)
        {
            _layers[_layers.Count - 1][i].CalculateError(desiredOutput[i]);
            error += _layers[_layers.Count - 1][i].GetError();
        }
        _errors.Add(error);
    }

    void BackPropagate()
    {
        for (int i = _layers.Count - 2; i >= 0; --i)
        {
            for (int j = 0; j < _layers[i].Count; ++j)
            {
                _layers[i][j].CalculateError(_layers[i + 1]);
                _layers[i][j].UpdateWeights(_layers[i + 1]);
            }
        }
    }

    public List<float> GetResults()
    {
        return _output;
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

    public void ResetError()
    {
        _errors.Clear();
    }

    public float GetError()
    {
        float sum = 0f;
        foreach(var err in _errors)
        {
            sum += err;
        }
        return sum / _errors.Count;
    }
}