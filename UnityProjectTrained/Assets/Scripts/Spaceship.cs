using System.Collections.Generic;
using UnityEngine;

public class Spaceship : MonoBehaviour
{
    public const float maxDistance = 300f;
    public Transform _planet;
    public bool Manual = false;

    NeuralNetwork _net = null;
    float lifetime = 0f;
    float elapsed = 0f;
    Vector3 _position = new Vector3();
    Vector3 _target = new Vector3();
    float _rotation;
    bool _training = false;
    bool _visualize;

    void Update()
    {
        if(!_visualize)
        {
            return;
        }
        transform.position = _position;
        transform.rotation = Quaternion.Euler(0f, 0f, _rotation);
    }

    public void UpdateDelta(float deltatime)
    {
        Vector3 deltaVector = _target - _position;
        float magnitude = deltaVector.magnitude;
        Vector3 directionToTarget = deltaVector / magnitude;
        Vector3 direction = new Vector3(Mathf.Cos(_rotation * Mathf.Deg2Rad), Mathf.Sin(_rotation * Mathf.Deg2Rad), 0f);

        float normalizedDistanceInv = 1f - Mathf.Min(magnitude, maxDistance) / maxDistance;// 0 ... 1
        float angleToTarget = Vector3.SignedAngle(direction, directionToTarget, new Vector3(0f, 0f, -1f)); //-180 ... 180
        float normalizedAngle = angleToTarget / 180f; // -1 ... 1

        var leftAngle = -Mathf.Min(normalizedAngle, 0f);// -1 .. 0 -> 0 ... 1
        var rightAngle = Mathf.Max(normalizedAngle, 0f);// 0 .. 1 -> 0 ... 1

        var inputs = new List<float>
        {
            normalizedDistanceInv,
            rightAngle,
            leftAngle
        };
        List<float> output;
        if(Manual)
        {
            output = new List<float>
            {
                Input.GetKey(KeyCode.W) ? 1f : 0f,
                Input.GetKey(KeyCode.D) ? 1f : 0f,
                Input.GetKey(KeyCode.A) ? 1f : 0f
            };
        }
        else if(_training)
        {
            output = new List<float>
            {
                normalizedDistanceInv < 0.99f ? 1f : 0f,
                rightAngle > 0.05f ? 1f : 0f,
                leftAngle > 0.05f ? 1f : 0f
            };
            _net.BackPropagate(inputs, output);
        }
        else
        {
            _net.FeedForward(inputs);
            output = _net.GetResults();
        }
        float moveForward = output[0];
        float moveRight = output[1];
        float moveLeft = output[2];
        
        float speed = 0f;
        if (moveForward > 0.9f)
        {
            speed += 10f;
        }
        float rotation = 0f;
        if (moveRight > .9f)
        {
            rotation -= 25f;
        }
        if (moveLeft > .9f)
        {
            rotation += 25f;
        }
        
        _rotation += rotation * deltatime;
        _position += direction * speed * deltatime;

        _position.x = Mathf.Clamp(_position.x, 0f, maxDistance);
        _position.y = Mathf.Clamp(_position.y, 0f, maxDistance);
        if(_rotation < 0f)
        {
            _rotation += 360f;
        }
        if(_rotation > 360f)
        {
            _rotation -= 360f;
        }

        elapsed = Mathf.Min(elapsed + deltatime, lifetime);
    }

    public void SetTraining(bool training)
    {
        _training = training;
    }

    public void Init(NeuralNetwork nn, Transform planet, float duration)
    {
        _net = nn;
        _planet = planet;
        lifetime = duration;
        elapsed = 0.0f;
        Reposition(planet.position.x, planet.position.y, transform.position.x, transform.position.y, transform.rotation.eulerAngles.z);
    }

    public void Reposition(float targetX, float targetY, float posX, float posY, float rotation, bool visualize = true)
    {
        _position.x = posX;
        _position.y = posY;
        _rotation = rotation;
        _visualize = visualize;
        _target.x = targetX;
        _target.y = targetY;
        gameObject.SetActive(visualize);
        Update();
    }
}
