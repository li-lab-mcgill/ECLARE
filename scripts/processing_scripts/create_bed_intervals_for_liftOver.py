import argparse
from eclare import return_setup_func_from_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    setup_func = return_setup_func_from_dataset(dataset_name)


    # Create bed intervals for liftOver
    setup_func(args, return_raw_data=True, return_type='data')
