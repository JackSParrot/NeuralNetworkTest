using System;
using UnityEngine;
using System.Collections.Generic;

public class Tester
{
	static Dictionary<List<float>, List<float>> TrainingData = new Dictionary<List<float>, List<float>>
		{
			{ new List<float>{ 0f, 0f, 0f }, new List<float>{ 0f } },
			{ new List<float>{ 0f, 0f, 1f }, new List<float>{ 1f } },
			{ new List<float>{ 0f, 1f, 0f }, new List<float>{ 1f } },
			{ new List<float>{ 0f, 1f, 1f }, new List<float>{ 0f } },
			{ new List<float>{ 1f, 0f, 0f }, new List<float>{ 1f } },
			{ new List<float>{ 1f, 0f, 1f }, new List<float>{ 0f } },
			{ new List<float>{ 1f, 1f, 0f }, new List<float>{ 0f } },
			{ new List<float>{ 1f, 1f, 1f }, new List<float>{ 1f } }
		};

	static NeuralNetwork net;

	public static void Train(NeuralNetwork net, List<float> inputs, List<float> desired)
	{
		net.FeedForward(inputs);
        var res = net.GetResults()[0];
        net.BackPropagation(desired);
		Debug.Log(string.Format("{0}, {1}, {2} -> {3} : {4} Error: ", inputs[0].ToString(), inputs[1].ToString(), inputs[2].ToString(), desired[0].ToString(), res) + net.GetError());
	}

	public static void FeedAndPrint(NeuralNetwork net, List<float> inputs)
	{
		net.FeedForward(inputs);
		var res = net.GetResults();
		Debug.Log(string.Format("{0}, {1}, {2} -> {3}", inputs[0].ToString(), inputs[1].ToString(), inputs[2].ToString(), res[0].ToString()));
	}

	public static void Init()
	{
        net = new NeuralNetwork(new List<int> { 3, 5, 1 }); //intiilize network
	}

    public static void Test()
	{
		//Itterate 5000 times and train each possible output
		//5000*8 = 40000 traning operations
		for (int i = 0; i < 5000; i++)
		{
			foreach (var kvp in TrainingData)
			{
				Train(net, kvp.Key, kvp.Value);
			}
		}
		//output to see if the network has learnt
		//WHICH IT HAS!!!!!
		foreach (var kvp in TrainingData)
		{
			FeedAndPrint(net, kvp.Key);
		}
	}

    public static void TrainStep()
	{
		foreach (var kvp in TrainingData)
		{
			Train(net, kvp.Key, kvp.Value);
		}
	}

    public static void TestResults()
	{
		foreach (var kvp in TrainingData)
		{
			FeedAndPrint(net, kvp.Key);
		}
	}
}
