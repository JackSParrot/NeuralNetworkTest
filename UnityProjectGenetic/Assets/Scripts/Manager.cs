using System.Collections.Generic;
using UnityEngine;

public static class CustomRandom
{
    public static System.Random rng = new System.Random();
}

public class Manager : MonoBehaviour
{
    public GameObject SpaceshipPrefab;
    public GameObject Planet;
    public bool Visualize = true;
    public int PopulationSize = 100;
    public float MatchTime = 20f;
    public int Timescale = 100;
    List<int> _topology = new List<int>{ 3, 3, 2 }; 

    List<NeuralNetwork> _nets = new List<NeuralNetwork>();
    List<Spaceship> _spaceships = new List<Spaceship>();
    List<KeyValuePair<int, float>> _weightedIndexes = new List<KeyValuePair<int, float>>();

    int _generationNumber = 0;
    float _elapsed = 0f;

    private void Start()
    {
        InitNeuralNetworks();
    }

    void Update ()
    {
        float dt = 0.02f;
        for (int t = 0; t < Timescale; ++t)
        {
            for (int i = 0; i < _spaceships.Count; ++i)
            {
                _spaceships[i].UpdateDelta(dt);
            }
            _elapsed += dt;
            if (_elapsed >= MatchTime)
                break;
        }

        if (_elapsed >= MatchTime)
        {
            _elapsed = 0f;
            ResetStatus();
        }
    }

    private void ResetStatus()
    {
        if(_nets.Count < 1)
		{
			return;
		}
        _nets.Sort((a, b) => a.GetFitness() > b.GetFitness() ? -1 : 1);
        float sum = 0.0f;
        for (int i = 0; i < _nets.Count; ++i)
        {
            sum += _nets[i].GetFitness();
        }
        int avg = (int)sum / _nets.Count;
        Debug.Log("generation: " + _generationNumber + " worst: " + (int)_nets[_nets.Count - 1].GetFitness() + " average: " + avg + " best: " + (int)_nets[0].GetFitness());

        _weightedIndexes.Clear();
        for (int i = 0; i < PopulationSize - 1; ++i)
        {
            _weightedIndexes.Add(new KeyValuePair<int, float>(i, _nets[i].GetFitness()));
        }
        _weightedIndexes.Sort((a, b) => 1 - 2 * Random.Range(0, 1));

        for (int i = 4; i < PopulationSize / 4; ++i)
        {
            _nets[i].CrossOver(_nets[GetRandomItemWithWeight()]);
        }

        for (int i = 0; i < PopulationSize / 4; i++)
        {
            int offset = PopulationSize / 4;
            _nets[i].SetFitness(0);

            _nets[i + offset].CloneValues(_nets[i]);
            _nets[i + offset].Mutate();
            _nets[i + offset].SetFitness(0f);

            _nets[i + offset * 2].CloneValues(_nets[i]);
            _nets[i + offset * 2].Mutate();
            _nets[i + offset * 2].SetFitness(0f);

            _nets[i + offset * 3].CloneValues(_nets[i]);
            _nets[i + offset * 3].Mutate();
            _nets[i + offset * 3].SetFitness(0f);

        }

        _generationNumber++;
        ResetSpaceships();
    }

    void ResetSpaceships()
    {
        for (int i = 0; i < PopulationSize; i++)
        {
            _spaceships[i].Reposition(Planet.transform.position.x,
                                      Planet.transform.position.y,
                                      Random.Range(Spaceship.maxDistance * 0.75f, Spaceship.maxDistance * 0.9f),
                                      Random.Range(Spaceship.maxDistance * 0.1f, Spaceship.maxDistance * 0.9f),
                                      Random.Range(0f, 360f),
                                      Visualize);
        }
    }

    void InitNeuralNetworks()
    {
        if (PopulationSize % 2 != 0)
        {
            PopulationSize = 4; 
        }

        _nets.Clear();
        for (int i = 0; i < PopulationSize; i++)
        {
            NeuralNetwork net = new NeuralNetwork(_topology);
            _nets.Add(net);
        }

        _spaceships.Clear();
        for (int i = 0; i < PopulationSize; i++)
        {
            Spaceship spaceship = Instantiate(SpaceshipPrefab).GetComponent<Spaceship>();
            spaceship.Init(_nets[i], Planet.transform, MatchTime);
            _spaceships.Add(spaceship);
        }
        ResetSpaceships();
    }

    int GetRandomItemWithWeight()
    {
        float max = 0.0f;
        for (int i = 0; i < _weightedIndexes.Count; ++i)
        {
            max += _weightedIndexes[i].Value;
        }

        float weight = Random.Range(0f, max);
        for (int i = 0; i < _weightedIndexes.Count; ++i)
        {
            weight -= _weightedIndexes[i].Value;
            if (weight <= 0f)
            {
                var item = _weightedIndexes[i];
                _weightedIndexes.RemoveAt(i);
                return item.Key;
            }
        }
        return -1;
    }
}
