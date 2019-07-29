using System;
using System.Collections.Generic;
using Layer = System.Collections.Generic.List<Neuron>;

public class Neuron
{
    static float TransferFunction(float value)
    {
        return (float)(1.0 / (1.0 + Math.Exp(-value)));
    }

    static float TransferFunctionDerivative(float value)
    {
        return value * (1.0f - value);
        //return TransferFunction(value) * (1.0f - TransferFunction(value));
        //return (float)Math.Exp(-value) / (float)Math.Pow(1.0 + Math.Exp(-value), 2.0);
    }

    /*static float TransferFunction(float value)
    {
        return (float)Math.Max(0f, value);
    }

    static float TransferFunctionDerivative(float value)
    {
        return value < 0f ? 0f : 1f;
    }*/

    static float mutationRate = 0.5f;
    private int _myIndex = 0;
    private float _outputVal = 0.0f;
    private float _outputDerivative = 0.0f;
    private float _error = 0.0f;
    private float _bias = 0.0f;
    private List<float> _outputWeights = new List<float>();

    public Neuron(int numOutputs, int myIndex)
    {
        _myIndex = myIndex;
        for (int c = 0; c < numOutputs; ++c)
        {
            _outputWeights.Add((float)CustomRandom.rng.NextDouble());
        }
        _bias = numOutputs > 0 ? (float)CustomRandom.rng.NextDouble() : 0f;
    }

    public void CloneValues(Neuron other)
    {
        _myIndex = other._myIndex;
        _outputWeights.Clear();
        foreach (var conn in other._outputWeights)
        {
            _outputWeights.Add(conn);
        }
        _bias = other._bias;
    }

    public float GetOutputValue() 
    {
        return _outputVal;
    }

    public float GetOutputDerivative()
    {
        return _outputDerivative;
    }

    public void SetOutputValue(float value)
    {
        _outputVal = value;
        _bias = 0f;
        for(int i = 0; i < _outputWeights.Count; ++i)
        {
            _outputWeights[i] = 1f;
        }
    }

    private float SumContributions(Layer nextLayer)
    {
        float sum = 0f;
        for (int n = 0; n < nextLayer.Count; ++n)
        {
            sum += _outputWeights[n] * nextLayer[n]._error * GetOutputDerivative();
        }
        return sum;
    }

    public void CalcError(Layer nextLayer)
    {
        _error = SumContributions(nextLayer);
    }

    public void CalcError(float targetVal)
    {
        _error = GetOutputDerivative() * (targetVal - GetOutputValue());
    }

    public void UpdateInputWeights(Layer prevLayer)
    {
        foreach (var neuron in prevLayer)
        {
            float newDeltaWeight =
                    neuron.GetOutputValue()
                    * _error;
            neuron._outputWeights[_myIndex] += newDeltaWeight;
        }
        _bias += _error;
    }

    public void FeedForward(Layer prevLayer)
    {
        float sum = _bias;
        foreach (var neuron in prevLayer)
        {
            float val = neuron.GetOutputValue();
            float weight = neuron._outputWeights[_myIndex];
            sum += val * weight;
        }
        _outputVal = TransferFunction(sum);
        _outputDerivative = TransferFunctionDerivative(_outputVal);
    }

    float GetMutation(float prev)
    {
        float factor = 1f;
        float rand = (float)CustomRandom.rng.NextDouble();
        if(rand < 0.2f)
        {
            factor = (float)CustomRandom.rng.NextDouble();
        }
        else if (rand < 0.4f)
        {
            factor = 1f - prev;
        }
        else if (rand < 0.6f)
        {
            factor = 0.75f + 0.25f * prev;
        }
        else if (rand < 0.8f)
        {
            factor = 0.25f + 0.75f * prev;
        }
        else
        {
            factor = rand * prev;
        }
        return factor;
    }

    public void Mutate()
    {
        for (int c = 0; c < _outputWeights.Count; ++c)
        {
            if ((float)CustomRandom.rng.NextDouble() * 100f <= mutationRate)
            {
                _outputWeights[c] = GetMutation(_outputWeights[c]);
            }
        }

        if ((float)CustomRandom.rng.NextDouble() * 100f <= mutationRate)
        {
            _bias = GetMutation(_bias);
        }
    }

    public void CrossOver(Neuron otherNeuron)
    {
        for(int c = 0; c < _outputWeights.Count; ++c)
        {
            if ((float)CustomRandom.rng.NextDouble() > 0.5f)
            {
                _outputWeights[c] = otherNeuron._outputWeights[c];
            }
            else
            {
                otherNeuron._outputWeights[c] = _outputWeights[c];
            }
        }
    }
}