import torch
import torch.nn as nn

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size = 3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size = 3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
            #nn.Dropout(p=0.1)
             # cuts the height and width from feature maps in half
            
        )
        self.Conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size = 3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
            #nn.Dropout(p=0.1)
             # 
            
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
            #nn.Dropout(p=0.1)
             #
            
        )
        self.Conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
            
            #nn.Dropout(p=0.1)
             #feature maps have size 14x14 means 14x14x128 
            
        )
        self.Conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2)
            
            #nn.Dropout(p=0.1)
             # Feature maps have size 7x7 means 7x7x256
            
        )
       
        self.head = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(in_features= 12544, out_features= 2048), 
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace = True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
            
            
                # 50 classes that need to be predicted
            
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.Conv_3(x)
        x = self.Conv_4(x)
        x = self.Conv_5(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x





######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
