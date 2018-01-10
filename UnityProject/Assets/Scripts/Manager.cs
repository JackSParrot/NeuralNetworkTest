﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Manager : MonoBehaviour {

    public GameObject SpermazoonOrefab;
    public GameObject Ovum;

    private bool isTraning = false;
    private int populationSize = 50;
    private int generationNumber = 0;
    private List<int> layers = new List<int>{ 3, 30, 30, 3}; //3 input and 1 output
    private List<NeuralNetwork> nets;
    private bool leftMouseDown = false;
    private List<Spermatozoon> spermazoons = null;
    List<KeyValuePair<float, int>> _weightedIndexes = new List<KeyValuePair<float, int>>();

    public int timescale = 10;

    int GetRandomItemWithWeight(float weight)
    {
        for(int i = 0; i < _weightedIndexes.Count; ++i)
        {
            if(weight < _weightedIndexes[i].Key)
            {
                var item = _weightedIndexes[i];
                _weightedIndexes.RemoveAt(i);
                return item.Value;
            }
        }
        return -1;
    }

	void Update ()
    {
        if (isTraning == false)
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
                float accum = 0f;
                for(int i = 0; i < populationSize - 1; ++i)
                {
                    accum += nets[i].GetFitness();
                    _weightedIndexes.Add(new KeyValuePair<float, int>(accum, i));
                }

                while(_weightedIndexes.Count > populationSize * 3 / 4)
                {
                    int a = GetRandomItemWithWeight(Random.Range(0f, accum));
                    int b = GetRandomItemWithWeight(Random.Range(0f, accum));
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
            StartCoroutine(TimerCoroutine());
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
        }
    }
    
    IEnumerator TimerCoroutine()
    {
        yield return new WaitForSeconds(35f / timescale);
        isTraning = false;
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
            spermazoon.Init(nets[i], Ovum.transform);
            spermazoons.Add(spermazoon);
        }
    }
}