using System.Collections.Generic;
using Layer = System.Collections.Generic.List<Neuron>;

public class NeuralNetwork
{
    List<Layer> _layers = new List<Layer>();
    float _error = 0f;
    float _fitness = 0f;

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
}