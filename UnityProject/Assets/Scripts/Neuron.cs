using System;
using System.Collections.Generic;
using Layer = System.Collections.Generic.List<Neuron>;

public class Neuron
{
    struct Connection
    {
        public float weight;
        public float deltaWeight;
    }

    static float TransferFunction(float value)
    {
        // tanh - output range [-1.0 ... 1.0]
        return (float)Math.Tanh(value);
    }

    static float TransferFunctionDerivative(float value)
    {
        // tanh derivative (1 - tanh^2(x))
        return 1.0f - value * value;
    }
/*
    static float TransferFunction(float value)
    {
        return (float)Math.Max(0.0f, Math.Min(1.0f, value));
    }

    static float TransferFunctionDerivative(float value)
    {
        return value > 0f ? 1f : 0f;
    }*/

    static float eta = 0.0f;   // [0.0 ... 1.0] overall net training rate
    static float alpha = 0.0f; // [0.0 ...  n ] multiplier of the last weight change (momentum)
    static float mutationRate = 1.0f;
    private int _myIndex = 0;
    private float _outputVal = 0.0f;
    private float _gradient = 0.0f;
    private List<Connection> _outputWeights = new List<Connection>();

    public Neuron(int numOutputs, int myIndex)
    {
        _myIndex = myIndex;
        for (int c = 0; c < numOutputs; ++c)
        {
            var conn = new Connection();
            conn.weight = (float)UnityEngine.Random.Range(-1f, 1f);
            _outputWeights.Add(conn);
        }
    }

    public Neuron(Neuron other)
    {
        _myIndex = other._myIndex;
        for (int c = 0; c < other._outputWeights.Count; ++c)
        {
            var conn = new Connection();
            conn.weight = other._outputWeights[c].weight;
            _outputWeights.Add(conn);
        }
    }

    public void CloneValues(Neuron other)
    {
        _myIndex = other._myIndex;
        for (int c = 0; c < other._outputWeights.Count; ++c)
        {
            var conn = _outputWeights[c];
            conn.weight = other._outputWeights[c].weight;
            conn.deltaWeight = other._outputWeights[c].deltaWeight;
            _outputWeights[c] = conn;
        }
    }

    public float GetOutputValue() 
    {
        return _outputVal;
    }

    public void SetOutputValue(float value)
    {
        _outputVal = value;
    }

    private float SumDOW(Layer nextLayer)
    {
        float sum = 0.0f;
        //sum our contribution of the error at the nodes we feed
        for (int n = 0; n < nextLayer.Count - 1; ++n)
        {
            sum += _outputWeights[n].weight * nextLayer[n]._gradient;
        }
        return sum;
    }

    public void CalcHiddenGradients(Layer nextLayer)
    {
        float dow = SumDOW(nextLayer);
        _gradient = dow * TransferFunctionDerivative(_outputVal);
    }

    public void CalcOutputGradients(float targetVal)
    {
        float delta = targetVal - _outputVal;
        _gradient = delta * TransferFunctionDerivative(_outputVal);
    }

    public void UpdateInputWeights(Layer prevLayer)
    {
        //the weights to be updated are in the connection container
        //in the neurons in the preceding layer

        for (int n = 0; n < prevLayer.Count; ++n)
        {
            Neuron neuron = prevLayer[n];
            float oldDeltaWeight = neuron._outputWeights[_myIndex].deltaWeight;

            float newDeltaWeight =
                    // individual input, modified by the gradient and train rate
                    eta
                    * neuron.GetOutputValue()
                    * _gradient
                    // also add momentum = a fraction of the previous delta weight
                    * alpha
                    * oldDeltaWeight;
            var conn = neuron._outputWeights[_myIndex];
            conn.deltaWeight = newDeltaWeight;
            conn.weight += newDeltaWeight;
            neuron._outputWeights[_myIndex] = conn;
        }
    }

    public void FeedForward(Layer prevLayer)
    {
        float sum = 0.0f;
        for (int n = 0; n < prevLayer.Count; ++n)
        {
            float val = prevLayer[n].GetOutputValue();
            float weight = prevLayer[n]._outputWeights[_myIndex].weight;
            sum += val * weight;
        }
        _outputVal = TransferFunction(sum);
    }

    public void Mutate()
    {
        for(int c = 0; c < _outputWeights.Count; ++c)
        {
            //mutate weight value 
            float randomNumber = UnityEngine.Random.Range(0.0f, 100.0f);
            if (randomNumber <= mutationRate)
            {
                randomNumber = randomNumber / mutationRate;
                var conn = _outputWeights[c];
                float weight = conn.weight;
                if (randomNumber <= 0.2f)
                { //if 1
                  //flip sign of weight
                    weight *= -1f;
                }
                else if (randomNumber <= 0.4f)
                { //if 2
                  //pick random weight between -1 and 1
                    weight = UnityEngine.Random.Range(-1f, 1f);
                }
                else if (randomNumber <= 0.6f)
                { //if 3
                  //randomly increase by 0% to 100%
                    float factor = UnityEngine.Random.Range(0f, 1f) + 1f;
                    weight *= factor;
                }
                else if (randomNumber <= 0.8f)
                { //if 4
                  //randomly decrease by 0% to 100%
                    float factor = UnityEngine.Random.Range(0f, 1f);
                    weight *= factor;
                }
                else
                {
                    weight = UnityEngine.Random.Range(-0.5f, 0.5f);
                }
                conn.weight = weight;
                _outputWeights[c] = conn;
            }
        }
    }

    public void CrossOver(Neuron otherNeuron)
    {
        for(int c = 0; c < _outputWeights.Count; ++c)
        {
            if(UnityEngine.Random.Range(0,100) > 50)
            {
                var conn = _outputWeights[c];
                conn.weight = otherNeuron._outputWeights[c].weight;
                conn.deltaWeight = otherNeuron._outputWeights[c].deltaWeight;
                _outputWeights[c] = conn;
            }
            else
            {
                var conn = otherNeuron._outputWeights[c];
                conn.weight = _outputWeights[c].weight;
                conn.deltaWeight = _outputWeights[c].deltaWeight;
                otherNeuron._outputWeights[c] = conn;
            }
        }
    }
}