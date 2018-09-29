
import torch
import functions
import utilities
import argparse

def get_input_args():
    """Creates 6 arguments for the predict.py script to load from CLI """
    parser = argparse.ArgumentParser(description="Imageclassifier model trainer")
    parser.add_argument("image_path", help="<Path to the image for prediction>", type=str)
    parser.add_argument("checkpoint", help="<checkpoint of model to be used>", type=str)
    parser.add_argument("--top_k", help="<top k probabilities>", type=str, default=5)
    parser.add_argument("--category_name", help="<checkpoint of model to be used>", type=str, default='cat_to_name.json')
    parser.add_argument("--gpu", help="<use gpu or not", type=bool , default=True)

    args = parser.parse_args()
    return args



def main():
    
    # Get CLI arguments
    args = get_input_args()

    # Build model from checkpoint
    model = functions.load_checkpoint(args.checkpoint)

    # Get probabilities, labels and flower name from prediction function
    top_probs, top_labels,  top_flowers = functions.predict(args.image_path, model, args.category_name, args.gpu, args.top_k)

    # Print result
    for i in zip(top_probs, top_labels, top_flowers):
        print(i)


if __name__ == "__main__":
    main()
