# lstm_various

Simple test the LSTM model with and without support various sequence length

In train.py

> USE_VARIOUS_LENGTH_DATASET = True

Set this parameter for enable the dataset to generate sequences with various lengths 
(padded with zeroes).

> USE_VARIOUS_LENGTH_MODEL = True

Set this parameter for enable the model to support various sequence length

Below is the tested results:

> USE_VARIOUS_LENGTH_DATASET = False  
> USE_VARIOUS_LENGTH_MODEL = False  
> Train loss: 5.725839059778082e-06, eval loss: 5.871739328237406e-06

> USE_VARIOUS_LENGTH_DATASET = True  
> USE_VARIOUS_LENGTH_MODEL = False  
> Train loss: 0.20795379454890886, eval loss: 0.20914572477340698

> USE_VARIOUS_LENGTH_DATASET = True  
> USE_VARIOUS_LENGTH_MODEL = True  
> Train loss: 8.49963589644176e-06, eval loss: 7.915679285967295e-06


> USE_VARIOUS_LENGTH_DATASET = False  
> USE_VARIOUS_LENGTH_MODEL = True  
> Train loss: 2.675574345782176e-06, eval loss: 2.7122323823884167e-06
