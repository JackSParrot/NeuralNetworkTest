using System;
using System.Collections.Generic;
using Layer = System.Collections.Generic.List<Neuron>;

public class Neuron
{
    static float kMutationRate = 0.5f;// percentage

    private int _myIndex = 0;
    private float _outputVal = 0f;
    private List<float> _outputWeights = new List<float>();

    public Neuron(int numOutputs, int myIndex)
    {
        _myIndex = myIndex;
        for (int c = 0; c < numOutputs; ++c)
        {
            _outputWeights.Add((float)CustomRandom.rng.NextDouble());
        }
        _outputVal = 0f;
    }

    public void CloneValues(Neuron other)
    {
        _myIndex = other._myIndex;
        _outputWeights.Clear();
        foreach (var conn in other._outputWeights)
        {
            _outputWeights.Add(conn);
        }
        _outputVal = other._outputVal;
    }

    float TransferFunction(float value)
    {
        return (float)(1.0 / (1.0 + Math.Exp(-value)));
    }

    public float GetOutputValue() 
    {
        return _outputVal;
    }

    public void SetOutputValue(float value)
    {
        _outputVal = value;
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
        _outputVal = TransferFunction(sum);
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