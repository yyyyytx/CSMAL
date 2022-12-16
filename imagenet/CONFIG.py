config_imagenet = {
    'ds_name':'imagenet',
    'n_cls': 1000,
    'train_batch_size':128,
    'test_batch_size':512,
    'momentum':0.9,
    'lr':0.1,
    'weight_decay':1e-4,
    'n_epoch': 90,
    'milestones': [50, 65, 75],
    'depth': 28,
    'k': 2,
    'drop': 0.3,
    'n_initial': 122120,
    'n_select': 64060,
    'n_total':1281167,

    #VAAL
    'latent_dim':32,
    'beta':1,
    'num_adv_steps':1,
    'num_vae_steps':2,
    'adversary_param':1
}

if __name__ == '__main__':
    print('%d'%(config_imagenet['n_total'] * config_imagenet['n_epoch'] // config_imagenet['train_batch_size']))