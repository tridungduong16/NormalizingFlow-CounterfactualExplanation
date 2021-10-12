import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.helpers import load_configuration_from_yaml
from utils.data_catalog import DataCatalog, EncoderNormalizeDataCatalog, TensorDatasetTraning


class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.last_layer = nn.Linear(64, 1)
        self.dp = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.last_layer(x)
        x = self.sigmoid(x)
        return x

def train_predictive_model(train_loader, model):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in tqdm(range(epochs)):
        total = 0
        correct = 0
        running_loss = 0
        model.eval()
        for local_batch, local_labels in (train_loader):
            local_batch = local_batch.type(torch.FloatTensor).to(device)
            local_labels = local_labels.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            outputs = model(local_batch)
            predicted = torch.ge(outputs, 0.5).int()
            outputs = outputs.reshape(-1)
            loss = criterion(outputs, local_labels.detach())
            running_loss += loss
            total += local_labels.size(0)
            cor = torch.eq(predicted.reshape(-1), local_labels).int().sum()
            correct += cor
            loss.backward(retain_graph=True)
            optimizer.step()
        accuracy = correct / total
        epoch_loss = running_loss / total
        if epoch % 10 == 0:
            print("\n Epoch {}, Accuracy {:.4f}, Loss {:.4f}".format(epoch, accuracy, epoch_loss))
    return model

if __name__ == '__main__':
    DATA_NAME = 'adult'
    DATA_PATH = '/home/trduong/Data/fairCE/data/processed_adult.csv'
    CONFIG_PATH = "/home/trduong/Data/fairCE/src/carla/data_catalog.yaml"
    CONFIGURATION_FOR_PROJECT = "/home/trduong/Data/fairCE/configuration/project_configurations.yaml"

    configuration_for_proj = load_configuration_from_yaml(CONFIGURATION_FOR_PROJECT)
    data_catalog = DataCatalog(DATA_NAME, DATA_PATH, CONFIG_PATH)
    encoder_normalize_data_catalog = EncoderNormalizeDataCatalog(data_catalog)
    data_frame = encoder_normalize_data_catalog.data_frame
    target = encoder_normalize_data_catalog.target

    labels = data_frame[target]
    features = data_frame.drop(columns = [target], axis = 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = TensorDatasetTraning(features, labels)


    # data_frame = encoder_normalize_data_catalog.data_frame
    # target = encoder_normalize_data_catalog.target
    # label = data_frame[target].values
    # features = data_frame.drop([target], axis = 1)
    # train_data = torch.hstack((z_representation, label))
    # train_dataset = DatasetAdult(train_data)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
