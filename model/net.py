"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sklearn.metrics import precision_score,recall_score,f1_score

NUM_CLASSES = 6
# alexnet and vggnet use hard coded input dim for the last layer block
# for resnet, densenet, get the dim from the model
NUM_CLASSIFIER_FEATURES = 256 * 6 * 6

def model_factory_pretraining(model_name,params,pretrained=True):

    if 'alexnet' in model_name:
        model = models.alexnet(pretrained=pretrained)
        # TODO: think about a way to avoid the hardcode numbers
        # The reason is that alexnet, vggnet is followed by a sequential model while resnet, densenet by a linear layer
        num_ftrs = 256 * 6 * 6

    elif 'vggnet' in model_name:
        model = models.vgg19(pretrained=pretrained)
        num_ftrs = 512 * 7 * 7

    elif 'densenet' in model_name:
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features

    # use resnet as the default
    else:
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features

    # freeze layers
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        
    if 'resnet' in model_name:
        model.fc = torch.nn.Sequential(
                        nn.Linear(num_ftrs, 4096),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(4096, 256),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(256, NUM_CLASSES),
                        torch.nn.LogSoftmax(dim=1)) 
    elif 'vggnet' in model_name:
        model.classifier = torch.nn.Sequential(
                        nn.Linear(num_ftrs, 256),
                        nn.ReLU(True),
                        nn.Dropout(),
                        # nn.Linear(256, 100),
                        # nn.ReLU(True),
                        # nn.Dropout(),
                        nn.Linear(256, NUM_CLASSES),
                        torch.nn.LogSoftmax(dim=1)) 
    else:
        model.classifier = torch.nn.Sequential(
                        nn.Linear(num_ftrs, 4096),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(4096, 256),
                        nn.ReLU(True),
                        nn.Dropout(),
                        nn.Linear(256, NUM_CLASSES),
                        torch.nn.LogSoftmax(dim=1)) 

    return model


# obsolete code
def model_factory(model_name,params,pretrained=False):
    if 'alexnet' in model_name:
        # create alexnet model
        model = models.alexnet(pretrained=pretrained)
        # num_ftrs = model.classifier.in_features
        num_ftrs = NUM_CLASSIFIER_FEATURES
        # model_conv.classifier = torch.nn.Linear(num_ftrs, 6)
        model.classifier = torch.nn.Sequential(
                        torch.nn.Linear(num_ftrs, 256), 
                        torch.nn.ReLU(), 
                        # torch.nn.Dropout(0.4),
                        torch.nn.Linear(256, NUM_CLASSES),                   
                        torch.nn.LogSoftmax(dim=1)) 


        # model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    elif 'vggnet' in model_name:
        # create vggnet model
        model = models.vgg16(pretrained=pretrained)
        # num_ftrs = model.classifier.in_features
        num_ftrs = NUM_CLASSIFIER_FEATURES
        # model_conv.classifier = torch.nn.Linear(num_ftrs, 6)
        model.classifier = torch.nn.Sequential(
                        torch.nn.Linear(num_ftrs, 256), 
                        torch.nn.ReLU(), 
                        # torch.nn.Dropout(0.4),
                        torch.nn.Linear(256, NUM_CLASSES),                   
                        torch.nn.LogSoftmax(dim=1)) 

    elif 'resnet' in model_name:
        # create resnet model
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        # model_conv.fc = torch.nn.Linear(num_ftrs, 6)
        model.fc = torch.nn.Sequential(
                        torch.nn.Linear(num_ftrs, 256), 
                        torch.nn.ReLU(), 
                        # torch.nn.Dropout(0.4),
                        torch.nn.Linear(256, NUM_CLASSES),                   
                        torch.nn.LogSoftmax(dim=1)) 

    elif 'densenet' in model_name:
        # create densenet model
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        # model_conv.classifier = torch.nn.Linear(num_ftrs, 6)
        model.classifier = torch.nn.Sequential(
                        torch.nn.Linear(num_ftrs, 256), 
                        torch.nn.ReLU(), 
                        # torch.nn.Dropout(0.4),
                        torch.nn.Linear(256, NUM_CLASSES),                   
                        torch.nn.LogSoftmax(dim=1)) 
    elif 'transfer_learning' in model_name:
        pretrained =True
        # create densenet model
        model = models.resnet50(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model.fc.in_features
        # model_conv.fc = torch.nn.Linear(num_ftrs, 6)
        model.fc = torch.nn.Sequential(
                        torch.nn.Linear(num_ftrs, 256), 
                        torch.nn.ReLU(), 
                        # torch.nn.Dropout(0.4),
                        torch.nn.Linear(256, NUM_CLASSES),                   
                        torch.nn.LogSoftmax(dim=1)) 
    else:
        model=Net(params)

    return model

class DenseInceptionNet(nn.Module):
    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(32*32*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        s = s.view(-1, 32*32*self.num_channels*4)             # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)




class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(32*32*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        s = s.view(-1, 32*32*self.num_channels*4)             # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)

class Net_Tranfer_Learning(nn.Module):
    """
    """

    def __init__(self, extractor, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.dropout_rate = params.dropout_rate
        
            # extractor from pretrained model

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(32*32*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        import torchvision.models as models
        model_conv = models.resnet18(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 6)





def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def precision(outputs, labels):
    """
    Compute the precision, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)

    # NOTE: here we need compute precision for multiclass classification, so we need define a average strategy, including macro,micro, and weighted and sample
    # check for details https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    return precision_score(labels,outputs, average='weighted')

def recall(outputs, labels):
    """
    Compute the recall, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return recall_score(labels,outputs, average='weighted')

def f1score(outputs, labels):
    """
    Compute the f1score, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)

    # https://stats.stackexchange.com/questions/431022/error-while-performing-multiclass-classification-using-gridsearch-cv
    return f1_score(labels,outputs,average='weighted')

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1score': f1score,
    # could add more metrics such as accuracy for each token type
}
