using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spermatozoon : MonoBehaviour
{
    const float maxDistance = 90f;
    private bool initilized = false;
    private Transform ovum;

    private NeuralNetwork net;

    public void update(float deltatime)
    {
        if (initilized == true)
        {
            Vector2 deltaVector = (ovum.position - transform.position);
            float magnitude = deltaVector.magnitude;
            Vector2 direction = deltaVector / magnitude;

            float normalizedDistanceInv = ((1.0f - Mathf.Min(magnitude, maxDistance) / maxDistance) * .5f) + .5f;
            
            float rad = (Mathf.Atan2(direction.y, direction.x));
            float normalizedRadians = (rad / Mathf.PI);

            float currentRotation = transform.rotation.eulerAngles.z % 360;
            if(currentRotation < 0f)
            {
                currentRotation += 360f;
            }
            float currentRotationNormalized = currentRotation / 360f;
            currentRotationNormalized = (currentRotationNormalized - 0.5f) * 2f;

            var inputs = new List<float>
            {
                normalizedRadians,
                normalizedDistanceInv,
                currentRotationNormalized
            };
            net.FeedForward(inputs);

            List<float> output = net.GetResults();
            float moveRight = output[0] * .5f + .5f;
            float moveLeft = output[1] * .5f + .5f;
            float moveForward = output[2] * .5f + .5f;

            float speed = 0.1f;
            if(moveForward > 0.9f)
            {
                speed += 5f;
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

            transform.position = transform.position + (speed * transform.up * deltatime);
            transform.Rotate(new Vector3(0f, 0f, rotation));

            float fitness = normalizedDistanceInv * .5f + .5f;
            net.AddFitness(fitness * fitness);
        }
    }

    public void Init(NeuralNetwork net, Transform ovum)
    {
        this.ovum = ovum;
        this.net = net;
        initilized = true;
    }


}
