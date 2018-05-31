using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spermatozoon : MonoBehaviour
{
    const float maxDistance = 90f;
    private float lifetime = 0f;
    private float elapsed = 0f;
    private bool initilized = false;
    private Transform ovum;

    private NeuralNetwork net;

    public void update(float deltatime)
    {
        if (initilized == true)
        {
            Vector3 deltaVector = (ovum.position - transform.position);
            float magnitude = deltaVector.magnitude;
            Vector3 directionToTarget = deltaVector / magnitude;
            Vector3 direction = transform.up;

            float normalizedDistanceInv = ((1.0f - Mathf.Min(magnitude, maxDistance) / maxDistance) * 2f) - 1f;// -1 ... 1
            
            /*float rad = (Mathf.Atan2(directionToTarget.y, directionToTarget.x));
            float normalizedRadians = (rad / Mathf.PI);

            float currentRotation = transform.rotation.eulerAngles.z % 360;
            if(currentRotation < 0f)
            {
                currentRotation += 360f;
            }
            float currentRotationNormalized = currentRotation / 360f;
            currentRotationNormalized = (currentRotationNormalized - 0.5f) * 2f;
            */
            float cosToTarget = Vector3.Dot(direction, directionToTarget);// -1 ... 1

            var inputs = new List<float>
            {
                //normalizedRadians,
                normalizedDistanceInv,
                //currentRotationNormalized
                cosToTarget
            };
            net.FeedForward(inputs);

            List<float> output = net.GetResults();
            float moveRight = output[0] * .5f + .5f;
            float moveLeft = output[1] * .5f + .5f;
            float moveForward = output[2] * .5f + .5f;

            float speed = 0.001f;
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

            transform.position = transform.position + (direction * speed * deltatime);
            transform.Rotate(new Vector3(0f, 0f, rotation));

            elapsed = Mathf.Min(elapsed + deltatime, lifetime);

            float fitness = normalizedDistanceInv * .5f + .5f;
            fitness = fitness + fitness * ((lifetime - elapsed + 0.1f) / lifetime);
            fitness += normalizedDistanceInv * .5f + .5f;
            net.AddFitness(fitness);
        }
    }

    public void Init(NeuralNetwork net, Transform ovum, float duration)
    {
        this.ovum = ovum;
        this.net = net;
        initilized = true;
        lifetime = duration;
        elapsed = 0.0f;
    }


}
