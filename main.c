#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define INPUT_SIZE 3
#define OUTPUT_SIZE 1
#define HIDDEN_SIZE 4
#define LEARNING_RATE .0001
#define EPOCHS 1000

#define min(a, b) (((double)(a) < (double)(b)) ? (double)(a) : (double)(b))
#define max(a, b) (((double)(a) > (double)(b)) ? (double)(a) : (double)(b))

typedef struct
{

    double weights_IH[INPUT_SIZE][HIDDEN_SIZE];
    double weights_HO[HIDDEN_SIZE][OUTPUT_SIZE];
    double bias_IH[HIDDEN_SIZE];
    double bias_HO[OUTPUT_SIZE];
    double hidden_net[HIDDEN_SIZE];
    double hidden_out[HIDDEN_SIZE];
    double out_net[OUTPUT_SIZE];
    double out_out[OUTPUT_SIZE];
} NeuralNetwork;

double relu(double x)
{
    return x < 0 ? 0 : x;
}
double derv_relu(double x)
{
    return x > 0 ? 1 : 0;
}

// Feed Forward
// - initialize random weights
// Back propagation
void init_n(NeuralNetwork *NN)
{

    // initialize weights of i_h
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            NN->weights_IH[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
        NN->bias_IH[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // initialize weights if h_o
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            NN->weights_HO[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
        NN->bias_HO[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

void feed_forward(NeuralNetwork *NN, double input[INPUT_SIZE])
{

    // feed hidden layer
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        NN->hidden_net[i] = NN->bias_IH[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            NN->hidden_net[i] += NN->weights_IH[i][j] * input[j];
        }
        NN->hidden_out[i] = relu(NN->hidden_net[i]);
    }

    // feed output layer

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        NN->out_net[i] = NN->bias_HO[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            NN->out_net[i] += NN->weights_HO[i][j] * NN->hidden_out[j];
        }
        NN->out_out[i] = relu(NN->out_net[i]);
    }
}

void back_propagation(NeuralNetwork *NN, double input[INPUT_SIZE], double target[OUTPUT_SIZE])
{
    // output_output
    double output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        output_error[i] = NN->out_out[i] - target[i];
    }
    // output_net
    // partial E / partial o_net = (partial E / partial o_out)  [output error] * derv(o_net)
    // for (int i = 0; i < OUTPUT_SIZE; i++)
    // {
    //     output_error[i] *= derv_relu(NN->out_net[i]);
    // }
    // till now output_error = partial E / partial o_net

    // hidden_out
    // partial E / partial hout = (partial E / partial o_net) [output_error] * (partial o_net / partial hout) [weight]
    double hidden_error[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_error[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            hidden_error[i] += output_error[j] * NN->weights_HO[i][j];
        }
    }

    // hidden_net
    // partial E / partial hnet = (partial E / partial hout) [hidden error] * (partial hout / partial hnet) [derv]
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        hidden_error[i] *= derv_relu(NN->hidden_net[i]);
    }

    // Updating weights
    // newW = w - partial E / partial w
    // partial E / partial w = (partial E / partial o_net) [output error] * (partial o_net / partial w) [hout]

    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            NN->weights_HO[i][j] -= LEARNING_RATE * output_error[j] * NN->hidden_out[i];
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        NN->bias_HO[i] -= LEARNING_RATE * output_error[i];
    }

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            NN->weights_IH[i][j] -= LEARNING_RATE * hidden_error[j] * input[i];
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        NN->bias_IH[i] -= LEARNING_RATE * hidden_error[i];
    }
}
double minn[INPUT_SIZE], maxx[INPUT_SIZE];
void do_preprocessing(double data[][INPUT_SIZE], int num_samples)
{

    for (int j = 0; j < INPUT_SIZE; j++)
        minn[j] = 100000.0, maxx[j] = -1.0;
    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            minn[j] = min(minn[j], data[i][j]);
            maxx[j] = max(maxx[j], data[i][j]);
        }
    }
    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            data[i][j] = (data[i][j] - minn[j]) / (maxx[j] - minn[j]);
        }
    }
}

void train(NeuralNetwork *NN, double data[][INPUT_SIZE], double target[][OUTPUT_SIZE], int num_samples)
{
    for (int i = 0; i < EPOCHS; i++)
    {
        double loss = 0.0;
        for (int j = 0; j < num_samples; j++)
        {
            feed_forward(NN, data[j]);

            for (int k = 0; k < OUTPUT_SIZE; k++)
            {
                double error = NN->out_out[k] - target[j][k];
                loss += error * error;
            }

            back_propagation(NN, data[j], target[j]);

            loss /= num_samples;
            if (i % 2 == 0)
            {
                printf("Epoch %d, Loss %.4f, \n", i, loss);
            }
        }
    }
}

void evaluate_train(NeuralNetwork *NN, double input[][INPUT_SIZE], double target[][OUTPUT_SIZE], int num_samples, double target_min, double target_max)
{

    printf("=== Train Prediction ===\n");

    for (int i = 0; i < num_samples; i++)
    {
        feed_forward(NN, input[i]);
        double pred_out = NN->out_out[0] * (target_max - target_min) + target_min;
        double actual = target[i][0] * (target_max - target_min) + target_min;
        printf("Sample %d: Predicted: $%.2fk, Actual: $%.2fk \n", i + 1, pred_out, actual);
    }
}

void evaluate_test(NeuralNetwork *NN, double test[INPUT_SIZE], double target_min, double target_max)
{
    printf("=== Test Prediction ===\n");

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        test[i] = (test[i] - minn[i]) / (maxx[i] - minn[i]);
    }
    feed_forward(NN, test);
    double pred = NN->out_out[0] * (target_max - target_min) + target_min;
    printf("Prediction: $%.2fk \n", pred);
}

int main()
{
    double training_inputs[][INPUT_SIZE] = {
        {3, 1500, 2.0}, {4, 2000, 1.5}, {2, 1000, 3.0}, {5, 2500, 1.0}, {3, 1800, 2.5}, {4, 2200, 1.2}, {2, 900, 4.0}, {6, 3000, 0.8}};

    double training_targets[][OUTPUT_SIZE] = {
        {300}, {400}, {200}, {500}, {350}, {450}, {180}, {600}};

    int num_samples = 8;

    do_preprocessing(training_inputs, num_samples);

    NeuralNetwork NN;
    init_n(&NN);

    printf("Training Neural Network\n");
    printf("Architecture: %d %d %d\n\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    double target_min = 1e9, target_max = -1;
    for (int i = 0; i < num_samples; i++)
    {
        target_min = min(target_min, training_targets[i][0]);
        target_max = max(target_max, training_targets[i][0]);
    }
    for (int i = 0; i < num_samples; i++)
    {
        training_targets[i][0] = (training_targets[i][0] - target_min) / (target_max - target_min);
    }

    train(&NN, training_inputs, training_targets, num_samples);

    evaluate_train(&NN, training_inputs, training_targets, num_samples, target_min, target_max);
    double test[INPUT_SIZE] = {3, 1600, 1.8};
    evaluate_test(&NN, test, target_min, target_max);

    return 0;
}