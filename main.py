from models import *
import wandb

def run_mae(args):
    # Data Loading
    ori_data = load_data(args)
    np.save(args.ori_data_dir, ori_data)  # save ori_data before normalization
    ori_data, min_val, max_val = min_max_scalar(ori_data)

    # Write statistics
    args.min_val = min_val
    args.max_val = max_val
    args.data_var = np.var(ori_data)
    print(f'{args.data_name} data variance is {args.data_var}')

    # Initialize the Model
    model = InterpoMAE(args, ori_data)
    wandb.login(key='023eab07f2db63032b62ece3605a0c0a8c3a62a2')

    # Khởi tạo Wandb run
    wandb.init(
        project= "ExtraMAE",
        name= "XAUUSD_M15",
        config=vars(args),
        reinit=True
    )
    if args.training:
        print(f'Start AutoEncoder Training! {args.ae_epochs} Epochs Needed.')
        model.train_ae()
        print(f'Start Embedding Training! {args.embed_epochs} Epochs Needed.')
        model.train_embed()
        print('Training Finished!\n')
    else:
        model = load_model(args, model)
        print(f'Successfully loaded the model!\n')



def main():
    home = os.getcwd()
    args = load_arguments(home)
    run_mae(args)


if __name__ == '__main__':
    main()
