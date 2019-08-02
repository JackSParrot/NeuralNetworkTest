using System.Collections.Generic;
using UnityEngine;

public class Spaceship : MonoBehaviour
{
    const float maxDistance = 90f;
    private float lifetime = 0f;
    private float elapsed = 0f;
    public Transform planet;

    private NeuralNetwork net;

    public void update(float deltatime)
    {
        Vector3 deltaVector = (planet.position - transform.position);
        float magnitude = deltaVector.magnitude;
        Vector3 directionToTarget = deltaVector / magnitude;
        Vector3 direction = transform.up;

        float normalizedDistanceInv = 1f - Mathf.Min(magnitude, maxDistance) / maxDistance;// 0 ... 1
        float angleToTarget = Vector3.SignedAngle(direction, directionToTarget, new Vector3(0f, 0f, -1f)); //-180 ... 180
        float normalizedAngle = angleToTarget / 180f; // -1 ... 1

        var leftAngle = -Mathf.Min(normalizedAngle, 0f);// -1 .. 0 -> 0 ... 1
        var rightAngle = Mathf.Max(normalizedAngle, 0f);// 0 .. 1 -> 0 ... 1

        var inputs = new List<float>
        {
            normalizedDistanceInv,
            leftAngle,
            rightAngle
        };
        net.FeedForward(inputs);

        List<float> output = net.GetResults();
        float moveForward = output[0];
        float moveLeft = output[1];
        float moveRight = output[2];

        float speed = 0f;
        if (moveForward > 0.9f)
        {
            speed += 10f;
        }
        float rotation = 0f;
        if (moveRight > .9f)
        {
            rotation += 2.5f;
        }
        if (moveLeft > .9f)
        {
            rotation -= 2.5f;
        }

        transform.position = transform.position + (direction * speed * deltatime);
        transform.rotation = Quaternion.Euler(0f, 0f, transform.rotation.eulerAngles.z + rotation);
        //transform.Rotate(new Vector3(0f, 0f, rotation));

        elapsed = Mathf.Min(elapsed + deltatime, lifetime);

        float fitness = normalizedDistanceInv * 2f;
        net.AddFitness(fitness * fitness);
    }

    public void Init(NeuralNetwork net, Transform planet, float duration)
    {
        this.planet = planet;
        this.net = net;
        lifetime = duration;
        elapsed = 0.0f;
    }
}
