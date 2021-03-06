import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',
                        type=str,
                        default='./dataset',
                        help='location of the noisy training images in numpy format', )
    parser.add_argument('--train_labels_path',
                        type=str,
                        default='dataset/train_clean.npy',
                        help='location of the clean target images in numpy format', )
    parser.add_argument('--test_data_path',
                        type=str,
                        default='data/test_noisy_100.npy',
                        help='location of the noisy test images in numpy format', )
    parser.add_argument('--background_path',
                        type=str,
                        default='data/train_background.npy',
                        help='location of the noisy background images in numpy format', )
    parser.add_argument('--predicted_test_label_save_dir',
                        type=str,
                        default="./predictions",
                        help='dir of the predicted labels named kaggle_prediction+time_stamp+.csv ', )
    parser.add_argument('--model_path',
                        type=str,
                        default="model/model.pth",
                        help='path to a output model dict dir', )
    parser.add_argument('--batch_size',
                        type=int,
                        default=64) # 64
    parser.add_argument('--validation_fraction',
                        type=float,
                        default=0.05) 
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0005)
    parser.add_argument('--bootstrap',
                        type=float,
                        default=1.0)
    parser.add_argument('--epochs',
                        type=int,
                        default=25)
    parser.add_argument('--random_seed',
                        type=int,
                        default=1337)
    parser.add_argument('--adversarial_weight',
                        type=float,
                        default=0.1)
    parser.add_argument('--beta',
                        type=float,
                        default=0.0025)


    

    args =   parser.parse_known_args()[0]
    print("-" * 25)
    print("Passed Arguments:")
    print("-" * 25)
    for k, v in vars(args).items():
        k, v = str(k), str(v)
        print('%s: %s' % (k, v))
    print("-" * 25)
    return args



