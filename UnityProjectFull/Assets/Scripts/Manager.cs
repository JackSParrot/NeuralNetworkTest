using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class CustomRandom
{
    public static System.Random rng = new System.Random();
}

public class Manager : MonoBehaviour {

    public GameObject SpermazoonOrefab;
    public GameObject Ovum;

    private bool isTraning = false;
    private int populationSize = 200;
    private int generationNumber = 0;
    private List<int> layers = new List<int>{ 2, 4, 4, 3}; //2 input and 1 output
    private List<NeuralNetwork> nets;
    private bool leftMouseDown = false;
    private List<Spermatozoon> spermazoons = null;
    List<KeyValuePair<int, float>> _weightedIndexes = new List<KeyValuePair<int, float>>();


    public float matchTime = 35f;
    public int timescale = 10;

    int GetRandomItemWithWeight()
    {
        float max = 0.0f;
        for(int i = 0; i < _weightedIndexes.Count; ++i)
        {
            max += _weightedIndexes[i].Value;
        }

        float weight = Random.Range(0f, max);
        for (int i = 0; i < _weightedIndexes.Count; ++i)
        {
            weight -= _weightedIndexes[i].Value;
            if(weight <= 0f)
            {
                var item = _weightedIndexes[i];
                _weightedIndexes.RemoveAt(i);
                return item.Key;
            }
        }
        return -1;
    }

    float _elapsed = 0f;
	void Update ()
    {
        if (!isTraning)
        {
            if (generationNumber == 0)
            {
                InitBoomerangNeuralNetworks();
            }
            else
            {
                nets.Sort();
                float sum = 0.0f;
                for (int i = 0; i < nets.Count; ++i)
                {
                    sum += nets[i].GetFitness();
                }
                int avg = (int)sum / nets.Count;
                UnityEngine.Debug.Log("generation: " + generationNumber + " worst: " + (int)nets[nets.Count - 1].GetFitness() + " average: " + avg + " best: " + (int)nets[0].GetFitness());

                _weightedIndexes.Clear();
                for(int i = 0; i < populationSize - 1; ++i)
                {
                    _weightedIndexes.Add(new KeyValuePair<int, float>(i, nets[i].GetFitness()));
                }

                while(_weightedIndexes.Count > populationSize * 0.75f)
                {
                    int a = GetRandomItemWithWeight();
                    int b = GetRandomItemWithWeight();
                    if(a >= 0 && b >= 0)
                    {
                        nets[a].CrossOver(nets[b]);
                    }
                }
                
                for (int i = 0; i < populationSize / 4; i++)
                {
                    int offset = populationSize / 4;

                    nets[i + offset].CloneValues(nets[i]);
                    nets[i + offset].Mutate();
                    nets[i + offset].SetFitness(0f);

                    nets[i + offset * 2].CloneValues(nets[i]);
                    nets[i + offset * 2].Mutate();
                    nets[i + offset * 2].SetFitness(0f);

                    nets[i + offset * 3].CloneValues(nets[i]);
                    nets[i + offset * 3].Mutate();
                    nets[i + offset * 3].SetFitness(0f);

                    nets[i].Mutate();
                    nets[i].SetFitness(0f);
                }
                System.GC.Collect();
            }
           
            generationNumber++;
            isTraning = true;
            CreateBoomerangBodies();
        }


        if (Input.GetMouseButtonDown(0))
        {
            leftMouseDown = true;
        }
        else if (Input.GetMouseButtonUp(0))
        {
            leftMouseDown = false;
        }

        if(leftMouseDown == true)
        {
            Vector2 mousePosition = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            Ovum.transform.position = mousePosition;
        }

        for (int t = 0; t < timescale; ++t)
        {
            for (int i = 0; i < spermazoons.Count; ++i)
            {
                spermazoons[i].update(Time.deltaTime);
            }
            _elapsed += Time.deltaTime;
            if (_elapsed >= matchTime)
                break;
        }

        if(_elapsed >= matchTime)
        {
            isTraning = false;
            _elapsed = 0f;
        }
    }

    private void CreateBoomerangBodies()
    {
        for (int i = 0; i < populationSize; i++)
        {
            spermazoons[i].transform.position = new Vector3(UnityEngine.Random.Range(-20f, 20f), UnityEngine.Random.Range(-20f, 20f), 0);
            spermazoons[i].transform.rotation = Quaternion.Euler(new Vector3(0f, 0f, UnityEngine.Random.Range(0f, 360f)));
        }

    }

    void InitBoomerangNeuralNetworks()
    {
        if (populationSize % 2 != 0)
        {
            populationSize = 20; 
        }

        nets = new List<NeuralNetwork>();

        for (int i = 0; i < populationSize; i++)
        {
            NeuralNetwork net = new NeuralNetwork(layers);
            nets.Add(net);
        }

        spermazoons = new List<Spermatozoon>();

        for (int i = 0; i < populationSize; i++)
        {
            Spermatozoon spermazoon = Instantiate(SpermazoonOrefab).GetComponent<Spermatozoon>();
            spermazoon.Init(nets[i], Ovum.transform, matchTime);
            spermazoons.Add(spermazoon);
        }
    }
}
