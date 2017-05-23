# TV-script-generation-with-embedding-RNN-and-LSTM
This project generates TV scripts with Recurrent Neural Networks and LSTMs. This project is trained on a script of the famous American sitcom, The Simpsons. I have used recurrent nets because while training on huge data, recurrent nets actually predict the outcome a lot better than any normal machine learning models.

##### *** This project will throw errors if trained on CPU instead of GPU ***

![Terminal screen_error](https://github.com/Satyaki0924/TV-script-generation-with-embedding-RNN-and-LSTM/blob/master/res/tensorflow_error.png?raw=true "Terminalerror")

### This project is configured for Linux and uses python3
To run this project, open up your bash terminal and write

```
chmod -R 777 setup.sh
```
This will set up the project enviornment for you. This must be run with administrator rights.

```
./setup.sh
```

#### * Virtual enviornment will be setup for you
Install the required packages using the following command.

```
source venv/bin/activate
pip install -r requirements.txt
```

## Train the project

```
python run_me.py
```

![Terminal screen_1](https://github.com/Satyaki0924/TV-script-generation-with-embedding-RNN-and-LSTM/blob/master/res/training.png?raw=true "Terminal1")

### Example of bad training
Setting parameters wrong will lead to overfitting, underfitting or other errors. The following is an example of overfitting.

![Terminal screen_2](https://github.com/Satyaki0924/TV-script-generation-with-embedding-RNN-and-LSTM/blob/master/res/loss1.png?raw=true "Terminal2")

### Loss graph under correct training

![Terminal screen_3](https://github.com/Satyaki0924/TV-script-generation-with-embedding-RNN-and-LSTM/blob/master/res/loss2.png?raw=true "Terminal4")

## Test the project
Run the python file, following the instructions

```
python run_me.py
```

The outcome should look something like this:

![Terminal screen_4](https://github.com/Satyaki0924/TV-script-generation-with-embedding-RNN-and-LSTM/blob/master/res/testing.png?raw=true "Terminal4")


#### Author: Satyaki Sanyal
##### *** This project is strictly for educational purposes only. ***
