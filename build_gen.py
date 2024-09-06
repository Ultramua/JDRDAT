import svhn2mnist
from HGLEnet import *


def Generator(device,num_class, input_size, sampling_rate, num_T,
              out_graph, dropout_rate, pool, pool_step_rate):
    return HGLEnet(device,num_class, input_size, sampling_rate, num_T,
                   out_graph, dropout_rate, pool, pool_step_rate)

def Disentangler(in_feature,out_feature1,out_feature2):
    return svhn2mnist.Feature_disentangle(in_feature=in_feature,out_feature1=out_feature1,out_feature2=out_feature2)


def Classifier(in_features,out_features):
    return svhn2mnist.Predictor(in_features=in_features,out_features=out_features)


def Feature_Discriminator(in_features,out_features):
    return svhn2mnist.Feature_discriminator(in_features=in_features,out_features=out_features)


def Reconstructor(in_features,out_features):
    return svhn2mnist.Reconstructor(in_features=in_features,out_features=out_features)


def Mine(in_features,out_features):
    return svhn2mnist.Mine(in_features=in_features,out_features=out_features)
