
config_cifar10 = {
    'ds_name':'CIFAR10',
    'n_cls': 10,
    'train_batch_size':64,
    'test_batch_size':256,
    'momentum':0.9,
    'lr':0.1,
    'weight_decay':5e-4,
    'n_epoch': 200,
    'milestones': [150],
    'acc_epoch': 100,
    'mu': 7,
    'n_semi_iteration': 200
}


config_cifar100 = {
    'ds_name':'CIFAR100',
    'n_cls': 100,
    'train_batch_size':64,
    'test_batch_size':1024,
    'momentum':0.9,
    'lr':0.1,
    'weight_decay':5e-4,
    'n_epoch': 200,
    'milestones': [150],
    'acc_epoch': 100,
    'mu': 7,
    'n_semi_iteration': 200
}


