using System;
using System.Collections.Generic;
using Layer = System.Collections.Generic.List<Neuron>;

interface IActivationFunction
{
    float Activate(float value);
    float Derivative(float value);
}

class Sigmoid : IActivationFunction
{
    public float Activate(float value)
    {
        float k = (float)Math.Exp(value);
        return k / (1.0f + k);
    }
    public float Derivative(float value)
    {
        return value * (1f - value);
    }
}

public class Neuron
{
    const float kMutationRate = 0.5f;// percentage
    const float kLearningRate = 0.5f;

    int _myIndex = 0;
    float _output = 0f;
    float _derivativeOutput = 0f;
    float _error = 0f;
    private List<float> _outputWeights = new List<float>();
    IActivationFunction _transfer = new Sigmoid();

    public Neuron(int numOutputs, int myIndex)
    {
        _myIndex = myIndex;
        for (int c = 0; c < numOutputs; ++c)
        {
            _outputWeights.Add((float)CustomRandom.rng.NextDouble());
        }
        _output = 0f;
    }

    public void CloneValues(Neuron other)
    {
        _myIndex = other._myIndex;
        _outputWeights.Clear();
        foreach (var conn in other._outputWeights)
        {
            _outputWeights.Add(conn);
        }
        _output = other._output;
    }

    public float GetOutputValue() 
    {
        return _output;
    }

    public void SetOutputValue(float value)
    {
        _output = value;
    }

    public float GetError()
    {
        return _error;
    }

    public void FeedForward(Layer prevLayer)
    {
        float sum = 0f;
        foreach (var neuron in prevLayer)
        {
            float val = neuron.GetOutputValue();
            float weight = neuron._outputWeights[_myIndex];
            sum += val * weight;
        }
        _output = _transfer.Activate(sum);
    }

    public void CalculateError(float desiredOutput)
    {
        _derivativeOutput = _transfer.Derivative(_output);
        _error = (desiredOutput - _output) * _derivativeOutput;
    }

    public void CalculateError(Layer nextLayer)
    {
        _error = 0f;
        _derivativeOutput = _transfer.Derivative(_output);
        for (int i = 0; i < nextLayer.Count - 1; ++i)
        {
            var n = nextLayer[i];
            _error += _outputWeights[n._myIndex] * n._error;
        }
        _error *= _derivativeOutput;
    }

    public void UpdateWeights(Layer nextLayer)
    {
        for (int i = 0; i < nextLayer.Count - 1; ++i)
        {
            var n = nextLayer[i];
            _outputWeights[n._myIndex] += kLearningRate * n._error * _output;
        }
    }

    float GetMutation(float prev)
    {
        if ((float)CustomRandom.rng.NextDouble() * 100f > kMutationRate)
        {
            return prev;
        }
        float rand = (float)CustomRandom.rng.NextDouble();
        if(rand < 0.2f)
        {
            return (float)CustomRandom.rng.NextDouble();
        }
        else if (rand < 0.4f)
        {
            return 1f - prev;
        }
        else if (rand < 0.6f)
        {
            return 0.75f + 0.25f * prev;
        }
        else if (rand < 0.8f)
        {
            return 0.25f + 0.75f * prev;
        }
        return rand * prev;
    }

    public void Mutate()
    {
        for (int c = 0; c < _outputWeights.Count; ++c)
        {
            _outputWeights[c] = GetMutation(_outputWeights[c]);
        }
    }

    public void CrossOver(Neuron otherNeuron)
    {
        for(int c = 0; c < _outputWeights.Count; ++c)
        {
            if (CustomRandom.rng.NextDouble() > 0.75)
            {
                _outputWeights[c] = otherNeuron._outputWeights[c];
            }
        }
    }
}