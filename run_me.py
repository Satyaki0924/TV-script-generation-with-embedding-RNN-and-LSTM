"""
This is a project by Satyaki Sanyal.
This project must be used for educational purposes only.
Follow me on:
LinkedIn - https://www.linkedin.com/in/satyaki-sanyal-708424b7/
Github - https://github.com/Satyaki0924/
Researchgate - https://www.researchgate.net/profile/Satyaki_Sanyal
"""
from src.main import Main


def main():
    num_epochs = 70
    batch_size = 128
    rnn_size = 256
    embed_dim = 256
    seq_length = 15
    learning_rate = 0.01
    show_every_n_batches = 10
    while True:
        try:
            ip = int(input('Enter 1. to train model, 2. to print scripts, 3. Exit \n>> '))
            if ip == 1 or ip == 2:
                Main().assert_v(num_epochs, batch_size, rnn_size, embed_dim, seq_length, learning_rate,
                                show_every_n_batches, ip)
            if ip == 3:
                print('*** Thank you ***')
                break
            else:
                print('*** Input not recognized. Try Again! ***')
        except Exception as e:
            print('***** EXCEPTION FACED: ' + str(e) + ' *****')


if __name__ == '__main__':
    main()
