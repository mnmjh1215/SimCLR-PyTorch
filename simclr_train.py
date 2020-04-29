from simclr_trainer import SimCLRTrainer
from dataloader import load_cifar10_for_contrastive_learning
from models.resnet import ResNet18, ResNet50
from models.projection import ProjectionHead
from models.simclr_model import SimCLRModel


def train(args):
    print("Preparing Data")
    train_dataloader = load_cifar10_for_contrastive_learning(batch_size=args.batch_size)
    
    print("Creating Model")
    if args.model_type == 'ResNet18':
        resnet = ResNet18()
        projection = ProjectionHead(512, 512, 512)
    elif args.model_type == 'ResNet50':
        resnet = ResNet50()
        projection = ProjectionHead(2048, 2048, 2048)
    else:
        print("model_type must be one of ResNet18 or ResNet50")
        raise NotImplementedError
    model = SimCLRModel(resnet, projection)
    
    print("Creating Trainer")
    trainer = SimCLRTrainer(model, train_dataloader, 
                            learning_rate=args.learning_rate, weight_decay=args.weight_decay, temperature=args.temperature,
                            linear_warmup_epochs=args.linear_warmup_epochs, total_epochs=args.total_epochs,
                            print_interval=args.print_interval)
    
    print("Start Training...")
    trainer.train()
    
    
def get_args():
    import argparse

    argument_parser = argparse.ArgumentParser("Python script to train ResNet using SimCLR method on CIFAR10 dataset")
    
    argument_parser.add_argument("--model_type",
                                 choices=['ResNet18', 'ResNet50'],
                                 default='ResNet18')
    
    argument_parser.add_argument("--batch_size",
                                 default=512,
                                 type=int)
    argument_parser.add_argument("--learning_rate",
                                 default=3e-3,
                                 type=float)
    argument_parser.add_argument("--weight_decay",
                                 default=1e-6,
                                 type=float)
    argument_parser.add_argument("--temperature",
                                 default=0.5,
                                 type=float)
    argument_parser.add_argument("--linear_warmup_epochs",
                                 default=10,
                                 type=int)
    argument_parser.add_argument("--total_epochs",
                                 default=100,
                                 type=int)
    argument_parser.add_argument("--print_interval",
                                 default=10,
                                 type=int)
    
    args = argument_parser.parse_args()
    return args

    
if __name__ == '__main__':
    args = get_args()
    train(args)