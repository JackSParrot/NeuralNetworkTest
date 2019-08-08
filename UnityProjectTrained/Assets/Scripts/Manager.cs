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
    public bool Train = true;
    public bool Visualize = true;
    public int PopulationSize = 100;
    public float MatchTime = 20f;
    public int Timescale = 100;

    NeuralNetwork _net;
    Spaceship _spaceship;
    List<Spaceship> _spaceships = new List<Spaceship>();

    int _generationNumber = 0;
    float _elapsed = 0f;

    private void Start()
    {
        InitPopulation();
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
            ResetSpaceships();
        }
    }

    void ResetSpaceships()
    {
        for (int i = 0; i < PopulationSize; i++)
        {
            _spaceships[i].SetTraining(Train);
            _spaceships[i].Reposition(Planet.transform.position.x,
                                      Planet.transform.position.y,
                                      Random.Range(0f, Spaceship.maxDistance),
                                      Random.Range(0f, Spaceship.maxDistance),
                                      Random.Range(0f, 360f),
                                      Visualize);
        }
        if(Train)
        {
            Debug.Log("Error: " + _net.GetError());
            _net.ResetError();
        }
    }

    void InitPopulation()
    {
        _net = new NeuralNetwork(new List<int> { 3, 3, 3 });
        _spaceships.Clear();
        for (int i = 0; i < PopulationSize; i++)
        {
            Spaceship spaceship = Instantiate(SpaceshipPrefab).GetComponent<Spaceship>();
            spaceship.Init(_net, Planet.transform, MatchTime);
            _spaceships.Add(spaceship);
        }
        ResetSpaceships();
    }
}
