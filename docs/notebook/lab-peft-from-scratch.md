<h1 align="center">PEFT from Scratch</h1>
Questions
1. How to implement different PEFT approaches ?
2. How to update these large pre-trained models with low resources  ?
3. How to apply PEFT methods to large language models for optimization of the model performance efficiently ?


### Setup and Requirements



We will use the pretrained model t5-small or autoregressive models like gpt-neo-125M for running our experimentation faster.


```python
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import time 
from tqdm.notebook import tqdm 
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
from IPython.display import display, HTML

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import  AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import datasets
from peft import get_peft_model, LoraConfig, TaskType, PrefixTuningConfig, PromptEncoderConfig


```


```python
# parameters 
random_seed = 42
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = "google-t5/t5-small"
causal_model = "EleutherAI/gpt-neo-125m"

dataset = "stanfordnlp/imdb"

epochs = 3
batch_size = 8
learning_rate = 1e-4

max_length = 512

```

    cpu



```python
dataset= load_dataset('imdb', split=['train[45%:55%]', 'test[45%:55%]', 'unsupervised[45%:55%]'])
dataset = datasets.DatasetDict({
    "train":dataset[0],
    "test":dataset[1],
    "unsupervised":dataset[2]

})

classes = ["negative", "positive"]

dataset = dataset.map(
    lambda x : {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)
dataset["train"][0]
```


    Map:   0%|          | 0/2500 [00:00<?, ? examples/s]



    Map:   0%|          | 0/2500 [00:00<?, ? examples/s]



    Map:   0%|          | 0/5000 [00:00<?, ? examples/s]





    {'text': "Recap: Full moon. A creature, a huge werewolf, is on the hunt. Not for flesh, not for blood (not that it seem to mind to take a bite on the way though), but for a mate. He is on the hunt for a girl. Not any girl though. The Girl. The girl that is pure (and also a werewolf, although she doesn't know it yet). Three, well check that, two cops (after the first scene) and an old bag lady is all that can stop it, or even knows that the thing killing and eating a lot of folks around full moon is a werewolf. This particular powerful werewolf, Darkwolf, is closing in on the girl. If he gets her, mankind is doomed. Now the cops has to find the girl, convince her not only that there is someone, a werewolf nonetheless, that wants to rape her, and perhaps kill her, but that she is a werewolf herself. And then they got to stop him...<br /><br />Comments: This is one for the boys, the teenage boys. A lot of scenes with semi-nude girls more or less important for the plot. Mostly less. Well I guess you need something to fill some time because the plot is (expectedly) thin. And unfortunately there is little besides the girls to help the plot from breaking. One usually turns to two main themes. Nudity. Check. And then special effects. Hmm... Well there are some things that you might call effects. They're not very special though. In fact, to be blunt, they are very bad. The movie seems to be suffering of a lack of funds. They couldn't afford clothes for some of the girls ;), and the effects are cheap. Some of the transformations between werewolf and human form, obviously done by computer, are really bad. You might overlook such things. But the Darkwolf in itself is very crude too, and you never get to see any killings. Just some mutilated corpses afterwards. And there is surprisingly little blood about, in a movie that honestly should be drenched in blood.<br /><br />I'm not sure what to say about actors and characters. Most of the times they do well, but unfortunately there are lapses were the characters (or actors) just looses it. A few of these lapses could be connected with the problems mentioned above. Like the poor effects, or the poor budget(?). That could explain why there is precious little shooting, even if the characters are armed like a small army and the target is in plain sight (and not moving). But hey, when you're in real danger, there nothing that will save your life like a good one-liner...<br /><br />Unfortunately that can't explain moments when the Cop, Steve, the only one who knows how to maybe deal with the problem, the werewolf that is, runs away, when the only things he can be sure of, is that the werewolf is coming for the girl, who is just beside him now, and that he cannot let it have her. But sure, it let the makers stretch the ending a little more...<br /><br />But I wouldn't mind seeing none of the lead actors/actresses get another try in another movie.<br /><br />Well. To give a small conclusion: Not a movie that I recommend.<br /><br />3/10",
     'label': 0,
     'text_label': 'negative'}



## Problems with Fine-tuning
Fine-tuning large language models like GPT are expensive, present several significant challenges for the average user such as computational expense, storage requirements and operational flexibility when managing multiple fine-tuned models.
1. High computational requirements
2. Expensive storage for checkpoints
3. Slow task switching with multiple fine-tuned models
4. catastrophic forgetting
5. Quality and Representation of Fine-tuning Data


```python
import torch 
import numpy as np

```


```python
_ = torch.manual_seed(0)
# define a rank 2 matrix W of size 10 x 10
d, k = (10, 10)
W_r = 2
W = torch.randn(d, W_r) @ torch.randn(W_r, k)
print(W)
print(np.linalg.matrix_rank(W))
print(W.size())

# apply the singular value decomposition on W
# W = U x S x V^T
U, S, V = torch.svd(W)
# rank r factorization we only keep first r singular value from D and corresponding columns of S and V^T
U_r = U[:, :W_r]
S_r = torch.diag(S[:W_r])
V_r = V[:, :W_r].t()
print(S_r)
# Computing A = U_r x S_r and B = V_r
A = U_r @ S_r
B = V_r
print(f"shape of the A is : {A.shape}")
print(f"shape of the B is : {B.shape}")

# let's generate the random bias and input 
bias = torch.randn(d)
x = torch.randn(d)

# computing straight line
# y = W^Tx + bias
y = W @ x + bias
# compute y' = (A * B) x + bias
y_prime = (A @ B) @ x + bias

print(f"original y using W : \n {y}")
print(f"computed using BA: \n {y_prime}")

# total params elements
print(f"total params of W : {W.nelement()}")
print(f"total params of A and B: {B.nelement() + A.nelement()}")
```

    tensor([[-1.0797,  0.5545,  0.8058, -0.7140, -0.1518,  1.0773,  2.3690,  0.8486,
             -1.1825, -3.2632],
            [-0.3303,  0.2283,  0.4145, -0.1924, -0.0215,  0.3276,  0.7926,  0.2233,
             -0.3422, -0.9614],
            [-0.5256,  0.9864,  2.4447, -0.0290,  0.2305,  0.5000,  1.9831, -0.0311,
             -0.3369, -1.1376],
            [ 0.7900, -1.1336, -2.6746,  0.1988, -0.1982, -0.7634, -2.5763, -0.1696,
              0.6227,  1.9294],
            [ 0.1258,  0.1458,  0.5090,  0.1768,  0.1071, -0.1327, -0.0323, -0.2294,
              0.2079,  0.5128],
            [ 0.7697,  0.0050,  0.5725,  0.6870,  0.2783, -0.7818, -1.2253, -0.8533,
              0.9765,  2.5786],
            [ 1.4157, -0.7814, -1.2121,  0.9120,  0.1760, -1.4108, -3.1692, -1.0791,
              1.5325,  4.2447],
            [-0.0119,  0.6050,  1.7245,  0.2584,  0.2528, -0.0086,  0.7198, -0.3620,
              0.1865,  0.3410],
            [ 1.0485, -0.6394, -1.0715,  0.6485,  0.1046, -1.0427, -2.4174, -0.7615,
              1.1147,  3.1054],
            [ 0.9088,  0.1936,  1.2136,  0.8946,  0.4084, -0.9295, -1.2294, -1.1239,
              1.2155,  3.1628]])
    2
    torch.Size([10, 10])
    tensor([[11.3851,  0.0000],
            [ 0.0000,  4.8439]])
    shape of the A is : torch.Size([10, 2])
    shape of the B is : torch.Size([2, 10])
    original y using W : 
     tensor([ 7.2684e+00,  2.3162e+00,  7.7151e+00, -1.0446e+01, -8.1639e-03,
            -3.7270e+00, -1.1146e+01,  2.0207e+00, -9.6258e+00, -4.1163e+00])
    computed using BA: 
     tensor([ 7.2684e+00,  2.3162e+00,  7.7151e+00, -1.0446e+01, -8.1638e-03,
            -3.7270e+00, -1.1146e+01,  2.0207e+00, -9.6258e+00, -4.1163e+00])
    total params of W : 100
    total params of A and B: 40



```python
plt.plot(x, y)
plt.show()
```


    
![png](output_8_0.png)
    



```python
plt.plot(x, y_prime)
```




    [<matplotlib.lines.Line2D at 0x7f5099a09810>]




    
![png](output_9_1.png)
    





```python
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
```

1. Training a NNs to classification of MNIST digits 
2. Fine-tuning the NNs on particular digit on which it does't perform well


```python
# 1. data transformation pipeline 
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

# 2. loading the training dataset
mnist_train = datasets.MNIST(
    root='./dataset',
    train=True, 
    download=True,
    transform=transform
)

# creating the dataset loader for training
train_loader = torch.utils.data.DataLoader(
    mnist_train,
    batch_size=10,
    shuffle=True
)

# 3. loading the training dataset
mnist_test = datasets.MNIST(
    root='./dataset',
    train=False, 
    download=True,
    transform=transform
)

# creating the dataset loader for training
test_loader = torch.utils.data.DataLoader(
    mnist_test,
    batch_size=10,
    shuffle=True
)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)
```

    cpu



```python
# NNs model for mnist digit classification
class NNClassifier(nn.Module):
    def __init__(self, hiddend_size1=1000, hiddend_size2=2000):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, hiddend_size1)
        self.fc2 = nn.Linear(hiddend_size1, hiddend_size2)
        self.fc3 = nn.Linear(hiddend_size2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        # Returns a new tensor with the same data as the self tensor but of a different shape .
        x = img.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NNClassifier().to(device)
```


```python
print(list(model.parameters()))
```

    [Parameter containing:
    tensor([[-0.0217,  0.0324,  0.0245,  ...,  0.0251, -0.0153,  0.0165],
            [-0.0317, -0.0023,  0.0119,  ..., -0.0138, -0.0319, -0.0170],
            [-0.0331,  0.0294, -0.0160,  ..., -0.0044,  0.0223, -0.0177],
            ...,
            [-0.0239, -0.0275, -0.0241,  ..., -0.0318, -0.0332,  0.0283],
            [-0.0185, -0.0276, -0.0083,  ..., -0.0080,  0.0042,  0.0142],
            [-0.0156, -0.0187, -0.0276,  ..., -0.0276, -0.0033, -0.0178]],
           requires_grad=True), Parameter containing:
    tensor([-1.0320e-02, -6.5207e-03, -3.0425e-02,  3.1773e-02,  2.3648e-02,
             2.5797e-02, -5.3310e-03,  2.5184e-02,  2.8565e-02, -3.3112e-02,
            -1.9557e-02,  7.6884e-03,  2.5767e-02,  1.1022e-02,  1.7873e-02,
             3.6960e-03,  1.1287e-02, -5.3714e-03,  2.5341e-02,  6.2376e-03,
            -7.9755e-04, -1.4545e-02, -2.8868e-02,  3.0665e-02, -1.6053e-02,
             2.4567e-02, -3.2694e-02,  1.0266e-02, -1.8655e-02,  3.0986e-02,
            -1.3568e-02,  3.4876e-02, -2.9578e-02,  2.7216e-02, -7.4883e-03,
             3.3605e-02, -1.4872e-03, -2.3862e-02,  1.6431e-02,  3.0644e-03,
            -2.5285e-02, -3.3244e-02, -3.1994e-02,  1.5422e-02, -1.2993e-02,
             4.7316e-03, -7.7458e-03, -8.3012e-03,  2.4295e-02,  1.4463e-02,
             3.0909e-02,  2.0509e-02, -9.2036e-03, -1.5312e-02, -4.0725e-03,
            -2.4484e-02, -1.4136e-03,  3.1083e-02, -5.4835e-03, -2.4125e-02,
            -1.4210e-02,  3.0606e-02, -1.4151e-03, -2.4470e-02,  2.0178e-02,
            -1.6197e-02,  1.7226e-02, -2.5792e-02,  2.6234e-02, -3.2808e-02,
            -5.0729e-03, -2.6916e-02,  2.6938e-02, -2.9573e-02,  2.2938e-02,
            -1.9897e-02,  3.3008e-04,  5.4161e-03, -3.3391e-02,  2.9945e-02,
            -1.7166e-02, -6.3009e-03,  2.0346e-02, -1.4461e-02,  1.8482e-02,
             6.0949e-03, -7.5448e-03, -3.2886e-02,  2.1630e-02,  7.4188e-03,
            -3.4950e-02, -8.3634e-03, -1.6844e-02,  2.0370e-02, -1.2094e-02,
             3.1708e-02, -2.9223e-02,  1.4402e-02,  7.0865e-03, -1.5358e-02,
             1.1856e-02,  3.1122e-02,  1.5203e-02, -1.8215e-03, -1.2824e-02,
             1.0726e-02,  1.1917e-02, -1.0603e-02, -2.5398e-02,  1.4085e-02,
             3.3672e-02,  2.1310e-02, -2.9447e-02,  9.3373e-03, -1.7143e-02,
             1.7928e-02,  2.4010e-02,  3.1620e-03, -3.4073e-02, -1.7843e-04,
             2.1446e-02, -5.7481e-03, -5.3269e-03,  2.2068e-02, -1.4882e-02,
             2.1116e-02,  1.4975e-02,  3.0249e-02, -1.8176e-02, -2.0800e-02,
            -3.2717e-02, -1.3635e-02, -2.2269e-03, -2.3452e-02,  6.2030e-03,
            -3.7176e-03, -3.2942e-02,  1.6464e-02,  2.1308e-03,  9.1494e-03,
             3.0661e-02, -1.0450e-02,  3.5434e-02, -1.0500e-02, -1.3349e-02,
             6.4561e-03, -2.2488e-02, -1.9313e-02, -2.1507e-02,  2.9707e-02,
            -1.0514e-03, -2.2687e-02,  1.6495e-02, -7.2441e-03,  9.0183e-03,
             3.5323e-02,  2.3491e-02,  3.4227e-02, -3.5294e-02,  3.1196e-02,
            -3.0435e-02, -1.8756e-03,  3.1710e-02, -2.5314e-02,  1.9563e-02,
            -5.1431e-03, -2.2396e-02,  1.6726e-02,  2.7843e-02,  1.9943e-03,
             9.1732e-03, -2.1799e-03,  1.3088e-02,  1.4959e-02, -1.8214e-02,
            -2.5891e-02,  1.7134e-02, -1.3138e-02, -3.4261e-02, -2.4443e-03,
             1.6028e-02,  9.3183e-03,  2.9106e-03, -1.4481e-02,  3.0286e-02,
             8.1368e-03, -2.3945e-02,  1.9110e-02,  1.2722e-02, -2.6075e-03,
            -2.9834e-02,  1.8304e-02, -3.1546e-02,  2.7695e-02,  3.7760e-03,
             1.4670e-02, -6.9188e-03,  1.0402e-02, -9.1942e-03,  3.4181e-02,
             2.2536e-02, -3.1036e-02,  3.3324e-02,  8.9956e-03, -3.4604e-02,
             1.9387e-02,  1.8981e-02, -8.9051e-03,  3.0992e-02, -1.2215e-02,
             2.0103e-02,  2.2554e-02, -3.3443e-02,  3.0147e-02,  1.9108e-02,
            -2.4510e-02, -1.2634e-02, -2.1231e-02,  3.4975e-02,  2.0375e-02,
             7.2947e-03,  4.3169e-03, -1.2881e-02, -1.3409e-02, -2.1335e-02,
            -2.3977e-02,  2.3418e-02,  6.6731e-03,  3.4438e-02, -2.0304e-02,
             3.2077e-03, -3.0646e-02, -2.4091e-02,  1.9362e-02, -3.6200e-03,
            -2.1357e-02,  7.0748e-03,  2.3027e-02, -1.3675e-04,  1.2392e-02,
            -3.3867e-02, -2.4129e-02, -2.3819e-02,  2.6258e-02,  7.8028e-03,
            -1.3049e-02,  3.2535e-02,  1.7173e-02,  3.3982e-02,  1.2429e-02,
             3.4691e-03, -3.6540e-03,  2.7349e-02,  1.3865e-02, -3.0865e-02,
             2.5375e-04,  1.4413e-02,  1.8267e-02,  1.9683e-02,  2.0639e-02,
             3.5192e-02,  3.5667e-03,  3.1047e-02, -2.9158e-03,  1.9225e-02,
            -8.5528e-03,  3.1992e-02, -1.0818e-02, -2.1563e-02,  8.3119e-03,
            -1.7559e-02,  2.6640e-02, -5.2229e-03, -2.5551e-02,  4.8802e-03,
             1.4288e-02, -6.5380e-03,  2.4955e-02, -5.4183e-03, -1.4058e-02,
             2.6932e-03,  3.3208e-02, -6.2700e-03, -1.5788e-02, -1.7684e-02,
             2.9466e-02,  2.7513e-02,  2.9602e-02, -3.3953e-02,  3.5185e-02,
            -2.7161e-02,  2.4324e-02, -3.3283e-02,  2.2554e-02, -5.0453e-03,
             2.2500e-02,  2.1988e-02, -2.4596e-02, -1.3986e-02,  2.7671e-02,
            -3.4754e-03, -1.2286e-02,  1.4984e-02, -4.4056e-03, -8.6189e-03,
             2.7208e-02, -2.9294e-02,  1.6323e-02, -3.1894e-02, -3.9037e-03,
            -3.3732e-02,  1.0061e-02,  2.4544e-02, -6.7358e-03,  2.8664e-02,
             3.3990e-02,  2.9531e-02,  1.7360e-02, -3.0523e-03, -1.9472e-03,
             3.5627e-02,  3.3541e-02, -3.4791e-02, -1.1263e-02, -1.1480e-02,
            -2.3141e-02, -3.3590e-02,  1.1438e-02,  2.7258e-02,  2.4727e-02,
            -3.0529e-02, -1.0583e-02, -2.0665e-02,  3.1055e-02, -5.0806e-03,
            -2.0507e-02,  2.5286e-02, -1.3779e-03, -2.2157e-02,  6.1207e-03,
             3.2355e-02, -1.8250e-02, -3.1861e-02, -9.3196e-03,  2.1641e-02,
            -3.9089e-03,  3.4388e-03, -2.2020e-02, -4.4128e-03,  2.3871e-03,
             3.2001e-02, -2.5075e-03,  2.8722e-02, -4.2344e-03, -2.9219e-02,
             2.0086e-03,  1.5973e-02, -3.5049e-03,  2.1306e-02, -1.1166e-02,
             2.0079e-02, -6.7300e-03, -2.6517e-04, -1.1985e-02, -1.7145e-02,
             1.9042e-02,  2.2402e-02,  1.1510e-02, -2.1869e-02, -2.6135e-02,
            -2.9119e-03,  1.1450e-02,  1.6710e-02,  4.8621e-03,  1.9973e-02,
            -1.0327e-02, -2.5088e-02, -2.5622e-03, -3.3391e-02,  1.4364e-02,
            -8.9138e-03,  1.4032e-02,  2.5126e-02, -1.6650e-02, -1.9333e-02,
             2.2969e-03, -2.5163e-02, -2.5816e-02,  9.7195e-03,  1.2432e-02,
            -2.8958e-02,  6.3350e-03,  3.3329e-02,  2.0824e-02, -1.2263e-02,
            -1.0363e-02, -9.2727e-03, -2.2585e-02, -2.0626e-02,  1.9297e-02,
             7.4085e-03,  1.7134e-03, -1.3231e-02,  2.3401e-02,  1.3387e-02,
             3.1168e-02,  1.1417e-02,  1.5655e-02,  3.2483e-02, -2.0565e-02,
            -1.7921e-02,  8.7170e-03, -3.4215e-02, -2.3987e-02,  2.6542e-02,
            -3.4977e-02,  1.9865e-02, -8.3847e-03, -2.9331e-02,  1.5026e-02,
            -3.4773e-02,  2.6966e-02,  2.7601e-02, -1.2405e-02,  1.7106e-02,
            -3.0025e-03, -1.5765e-02, -2.2292e-02, -2.9382e-02,  1.4102e-02,
            -5.2560e-03,  2.1957e-03, -2.1462e-02, -1.4539e-02, -1.0104e-02,
            -2.3779e-02,  3.4817e-02,  3.5219e-02, -3.8881e-03, -6.7443e-03,
            -2.3840e-02,  2.8530e-02,  2.9175e-02,  1.3148e-02, -1.1325e-02,
            -1.1776e-02, -1.7442e-02, -1.2934e-02,  7.2415e-03, -1.9589e-02,
            -1.4543e-02,  9.6600e-03,  1.3898e-02, -2.8426e-02,  4.4072e-03,
             9.4081e-03,  1.6309e-03,  3.0097e-02, -1.1128e-02,  3.6528e-03,
            -9.2725e-03,  1.3974e-02, -2.6080e-02,  2.0842e-02, -2.1798e-04,
             1.6775e-02, -2.4995e-02, -2.8316e-02, -1.4493e-02, -2.8528e-02,
             2.5058e-02, -8.3217e-03, -3.3703e-02, -4.7767e-03, -2.3862e-02,
            -2.0058e-02,  3.2306e-02,  2.9167e-02, -1.4795e-02,  5.3299e-03,
            -2.5088e-02,  1.2088e-02, -2.5961e-02, -7.5826e-03, -3.0156e-02,
            -1.8749e-02, -2.6915e-02, -2.9662e-03, -8.0834e-03,  2.6074e-02,
             2.1337e-02,  3.3836e-02, -1.4253e-02,  2.5713e-02,  1.0277e-02,
             2.6883e-02,  2.5550e-02,  7.3263e-04, -2.1505e-02, -2.9476e-02,
             1.7300e-02, -9.2658e-04, -2.6885e-02,  3.1527e-02, -3.1505e-02,
             1.6972e-03,  3.9439e-03, -1.3458e-02,  2.0468e-02, -1.3039e-02,
            -2.6961e-02,  2.1360e-02, -1.9148e-02,  9.4469e-03,  3.8634e-03,
             1.5208e-02, -3.4737e-02,  2.8281e-02,  1.1064e-02,  8.2391e-03,
             3.2464e-02, -1.1658e-02,  1.3489e-02, -1.7283e-02,  2.2119e-02,
            -3.4424e-02,  2.5421e-02, -1.2789e-02,  2.9809e-02,  1.0789e-02,
             2.3867e-02, -3.5295e-02,  2.3215e-02,  1.9149e-02, -2.4315e-02,
            -2.5146e-02, -8.6107e-04, -5.0835e-03, -1.8575e-02, -2.6215e-02,
            -2.7810e-02,  2.5646e-02,  3.0134e-02, -3.0501e-02, -2.9124e-02,
            -8.9983e-03,  3.3040e-02,  3.3356e-02, -1.9703e-02,  1.0565e-02,
            -2.8867e-02, -1.3825e-02, -3.0619e-02,  2.5332e-02,  2.1241e-03,
            -3.4285e-02,  9.5266e-03, -9.6307e-03, -3.0483e-02, -8.1293e-04,
            -1.9215e-02, -2.6393e-02,  1.0839e-02,  1.7522e-02,  1.2732e-02,
             2.4161e-03,  3.2747e-03, -2.7306e-02, -1.6981e-02, -2.5289e-02,
             1.4391e-02,  1.5814e-02, -3.2952e-02,  8.5922e-03, -1.8989e-02,
             2.6121e-02,  2.6907e-02, -2.3459e-02, -1.6141e-02, -3.3371e-02,
            -2.6357e-03,  2.4567e-02, -7.7612e-03, -2.1023e-02,  2.5129e-02,
             2.1122e-02, -8.7298e-03, -1.4417e-02, -3.4422e-05, -3.5433e-03,
            -2.5052e-02, -6.2602e-03,  1.9767e-02,  2.8703e-02,  2.2393e-02,
            -2.6030e-02,  2.7101e-02,  2.2300e-02, -8.3532e-03, -2.9488e-02,
             1.3485e-02,  4.4590e-03,  2.7863e-02, -4.9552e-03, -1.2408e-02,
             2.1340e-02,  3.3019e-02, -2.0870e-02, -1.0793e-02, -1.7556e-02,
            -1.8153e-02, -2.9978e-02, -1.5054e-04, -7.9508e-03, -2.0383e-02,
             1.8043e-03, -2.5907e-02,  2.1147e-03, -5.1263e-03, -8.3881e-03,
            -9.3727e-04, -2.5783e-02, -6.2917e-03, -2.2286e-02,  7.0261e-03,
            -2.6878e-02, -3.3122e-02,  2.7937e-02,  1.0864e-02,  2.6594e-02,
            -1.1931e-02,  5.8170e-03,  2.6352e-02,  1.2630e-02, -9.7866e-03,
            -1.6081e-04, -1.4795e-03,  2.5824e-02,  2.9091e-02,  1.1263e-02,
             1.0342e-02,  8.4446e-03,  3.1705e-02,  1.8338e-02, -1.8277e-02,
             8.5916e-04, -2.4103e-02,  1.7859e-02, -9.5928e-03,  1.3742e-02,
            -8.9115e-03,  1.4277e-02,  3.5388e-02, -1.7607e-02, -1.4693e-02,
            -9.8005e-03,  4.3073e-05, -3.3865e-02,  1.0280e-02,  1.5807e-02,
             1.5070e-02, -1.6566e-02,  1.7564e-02, -2.7704e-02, -2.2590e-02,
             8.8178e-03, -1.7540e-02,  2.6702e-02, -3.1916e-03,  1.7622e-02,
            -7.6404e-03,  2.8830e-02, -1.3591e-02,  2.4009e-02,  1.5386e-02,
             6.9780e-03,  1.8959e-02,  8.2545e-03, -6.6080e-03, -3.2939e-02,
             2.8128e-02, -1.7068e-02,  5.5168e-04, -5.2053e-03, -2.9445e-02,
             3.1236e-02,  1.9677e-02, -4.3910e-03, -3.2076e-02,  1.0516e-02,
            -3.1377e-02,  1.9781e-02,  3.1435e-02,  2.1929e-02,  1.3153e-02,
            -2.6972e-02,  2.1385e-02,  3.4706e-02,  1.4306e-02, -5.6461e-03,
             2.7818e-02,  5.8553e-03,  1.9443e-02,  2.9906e-02, -2.3895e-02,
            -4.4953e-03,  3.4025e-02, -2.7327e-02,  7.5911e-03, -1.0696e-02,
            -1.5784e-02, -3.3610e-02,  2.1257e-02, -8.0741e-03, -2.5527e-02,
            -2.9305e-02, -6.2372e-03,  6.2000e-03, -6.2799e-03,  2.2755e-02,
            -3.3834e-03, -8.0613e-03, -1.8107e-02,  4.3210e-03, -6.5641e-03,
            -2.9937e-02, -1.0537e-02, -1.7394e-02,  3.1030e-02, -3.5578e-02,
            -7.2662e-03, -4.8714e-03, -2.2475e-02, -5.1476e-03, -1.6219e-02,
             1.8819e-03, -2.7253e-02,  2.8513e-02, -2.1133e-02,  2.4286e-02,
            -1.1441e-02,  1.7717e-02,  3.2361e-03,  2.7704e-02, -2.1761e-02,
            -2.2392e-02,  3.1089e-02,  2.9018e-02,  3.0272e-02, -6.8205e-03,
            -2.5453e-02,  2.3631e-02, -2.0008e-02, -1.3758e-02, -3.0179e-02,
             9.0209e-03,  3.5342e-02,  1.8764e-02, -2.9572e-03,  1.4037e-02,
            -2.5208e-02, -7.9729e-03, -3.3988e-02, -2.0911e-02,  2.1514e-02,
             3.1382e-02,  6.7151e-03, -9.3605e-03,  2.7419e-02,  6.9611e-03,
            -1.8635e-02,  3.1656e-02, -1.4793e-02,  2.3778e-03, -1.7655e-02,
            -3.4396e-02, -1.3742e-02,  3.0899e-02,  8.0773e-03,  2.6538e-02,
            -1.9777e-02, -9.0255e-03,  1.1360e-02,  1.3888e-02,  1.7534e-02,
             3.1802e-02, -3.1403e-02,  2.4403e-02, -9.1145e-03,  1.8790e-02,
            -1.3794e-02,  6.3013e-03, -9.9022e-04,  5.3809e-03, -4.6361e-04,
             1.2676e-02, -2.3639e-02,  3.1625e-02, -2.5981e-02, -3.2198e-02,
             1.5652e-02,  1.8372e-02, -2.6063e-02,  1.2784e-02, -8.7744e-03,
            -3.2363e-02, -1.1799e-02, -2.8773e-03,  3.6047e-03,  3.3369e-02,
             1.8367e-02, -4.9440e-03, -2.5065e-02, -2.4709e-02, -1.0674e-02,
            -2.2570e-02, -1.0224e-02, -1.2301e-02, -1.2572e-02,  6.1015e-03,
             2.5827e-02, -2.7389e-02,  8.3895e-03, -1.4418e-02,  2.3136e-02,
             1.8191e-03, -9.2161e-03, -1.1298e-02, -3.5630e-02, -9.3792e-03,
            -1.6237e-02, -1.5355e-02, -3.5711e-02,  4.5125e-03, -4.9696e-03,
             2.7394e-03, -7.9843e-03,  1.4771e-02, -2.9733e-02, -1.8549e-03,
             1.6829e-02,  3.3979e-02,  9.2009e-03, -1.2639e-02, -3.4870e-02,
            -2.8328e-02,  2.7494e-03, -3.5614e-02, -3.2945e-02,  3.1098e-03,
            -1.7553e-03, -2.7198e-02, -4.6992e-04,  2.2827e-02,  2.8319e-02,
            -5.5863e-03, -2.9454e-02, -1.0707e-02, -2.1160e-02,  6.5313e-03,
            -1.6470e-02,  1.3873e-02,  2.9606e-02, -1.9141e-02, -1.2932e-02,
            -1.1128e-02, -2.7259e-02, -8.5404e-03,  1.4063e-03, -1.5799e-02,
            -5.2531e-04,  6.8968e-03, -3.2139e-02, -2.1133e-02,  5.9622e-04,
            -2.4588e-02,  2.4882e-02, -2.0747e-02,  3.2070e-02, -1.7916e-02,
            -3.1254e-02,  7.4483e-03, -2.6131e-02,  2.1663e-02, -1.9201e-02,
             1.8842e-02, -7.4075e-03,  7.4798e-03, -2.9741e-02,  8.1354e-03,
            -2.7016e-02,  2.3997e-02,  2.4825e-02,  3.3629e-02,  2.5536e-02,
            -1.3840e-02, -2.8280e-02, -2.0921e-02,  9.3194e-04,  7.7197e-03,
            -1.5052e-02, -6.9257e-03,  3.1434e-02,  1.7373e-04,  1.4073e-02,
             2.0773e-02,  1.4547e-02, -8.5811e-03, -2.3977e-02, -3.3969e-02,
             3.0125e-02, -6.3217e-03, -2.1554e-02,  2.5343e-02, -2.9926e-02,
            -2.3997e-03,  3.9591e-03, -5.7528e-03,  3.4311e-02,  1.6726e-02,
             1.7948e-02, -3.3646e-02, -2.2943e-02, -1.5972e-02, -3.4631e-02,
             9.5554e-03, -1.9555e-02, -2.2307e-02, -7.5593e-03, -5.3825e-03,
            -1.4440e-02, -1.5396e-02,  2.6183e-02, -2.9610e-02,  1.3939e-02,
            -1.1712e-02, -1.1865e-02,  3.0449e-02,  2.1924e-02,  2.5383e-03,
            -1.2323e-02,  1.2927e-02, -1.1346e-02,  9.4067e-03,  7.2429e-03,
            -1.9180e-02, -3.7484e-03, -2.7377e-02, -1.2778e-02, -2.9293e-02,
             2.1381e-02,  2.7661e-02,  3.4644e-02, -1.4897e-02, -1.9537e-02,
             6.6081e-03,  1.4900e-02, -7.8494e-03,  1.8571e-02,  1.7224e-02,
            -1.4418e-02, -3.3899e-02, -2.1095e-02, -2.1899e-02, -1.3322e-02,
            -2.4567e-02, -1.7855e-02, -1.4534e-02,  1.5108e-02, -8.2170e-03,
            -2.1554e-02,  2.1186e-02,  8.7807e-03,  3.2396e-02, -5.4814e-03,
             1.1285e-02, -2.0407e-02,  1.3838e-02,  1.7412e-02,  6.5471e-04,
             2.2826e-02,  2.5913e-02, -1.3132e-02,  2.4417e-02,  3.7038e-03,
            -3.1471e-02,  3.5355e-02,  1.0973e-03,  5.5835e-03,  9.1416e-03,
             1.3847e-02,  2.7203e-02,  2.6725e-02,  2.8856e-02,  1.4015e-03,
             2.8019e-02, -2.5718e-02, -6.4135e-03,  1.3450e-02,  2.1297e-02],
           requires_grad=True), Parameter containing:
    tensor([[ 2.7064e-02, -1.3890e-02,  1.2426e-02,  ...,  1.6834e-02,
             -2.2114e-02, -1.2481e-03],
            [ 8.6761e-04, -2.9554e-02,  1.0315e-02,  ..., -1.0449e-03,
              2.7714e-02, -1.7915e-02],
            [-4.0925e-03,  1.9115e-02, -2.9762e-02,  ...,  1.9335e-02,
              2.7546e-02,  1.9274e-02],
            ...,
            [-1.8041e-02, -1.8621e-02,  1.5718e-02,  ..., -1.9459e-02,
             -1.5317e-02, -6.8762e-03],
            [ 2.0010e-03,  5.0789e-03, -6.4813e-03,  ..., -2.7187e-05,
              1.8645e-02,  3.6312e-03],
            [-1.2214e-02, -7.0958e-03,  2.7411e-02,  ...,  1.2113e-03,
             -3.0699e-02,  1.4975e-02]], requires_grad=True), Parameter containing:
    tensor([ 0.0277, -0.0068, -0.0157,  ..., -0.0305,  0.0262, -0.0049],
           requires_grad=True), Parameter containing:
    tensor([[ 0.0028, -0.0114,  0.0146,  ..., -0.0143,  0.0011,  0.0098],
            [-0.0015, -0.0114, -0.0172,  ..., -0.0144, -0.0206,  0.0201],
            [ 0.0171, -0.0212,  0.0021,  ..., -0.0209, -0.0052, -0.0055],
            ...,
            [-0.0084, -0.0177,  0.0097,  ..., -0.0002, -0.0125,  0.0055],
            [ 0.0217,  0.0042,  0.0052,  ..., -0.0189, -0.0043, -0.0173],
            [ 0.0069,  0.0023,  0.0138,  ...,  0.0218,  0.0053,  0.0042]],
           requires_grad=True), Parameter containing:
    tensor([-0.0203, -0.0220,  0.0044, -0.0117, -0.0069,  0.0038,  0.0209, -0.0037,
            -0.0203,  0.0065], requires_grad=True)]



```python
def training(train_loader, model, epochs, iter_limit=None):
    # loss function and optimizer
    loss_fun = nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=0.001)

    iter_total=0
    for epoch in range(epochs):
        model.train()
        
        loss_sum = 0
        iter_num = 0
        
        data_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        if iter_limit is not None:
            data_iter.total = iter_limit
        for data in data_iter:
            iter_num += 1
            iter_total +=1

            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            output = model(x.view(-1, 28*28))
            loss  = loss_fun(output, y)

            
            loss_sum += loss.item()

            avg_loss = loss_sum / iter_num
            
            data_iter.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if iter_limit is not None and iter_total >= iter_limit:
                return avg_loss
    return avg_loss
```


```python
training(train_loader, model, epochs=1)
```

    Epoch 1: 100%|██████████| 6000/6000 [01:43<00:00, 58.11it/s, loss=0.24] 





    0.24042963718108756




```python
# make a copy of the original model's weight, we use it to prive that LoRA fine-tuning does't alter the original weights
org_weights = {}
for name, param in model.named_parameters():
    org_weights[name] = param.clone().detach()
```


```python
# the performance of  NNClassifier model
# It poorly performed on digit 2
# so we will fine-tune the model for digit 2

def test():
    correct = 0
    total = 0
    wrong_counts = [0 for i in range(10)]

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data 
            x  = x.to(device)
            y = y.to(device)

            output = model(x.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                   correct +=1
                else:
                    wrong_counts[y[idx]] +=1
                total += 1
    print(f"Accuracy:{round(correct/total, 3)}")
    for i in range(len(wrong_counts)):
        print(f"Wrong count for digit {i}: {wrong_counts[i]}")

test()
```

    Testing: 100%|██████████| 1000/1000 [00:00<00:00, 1041.01it/s]

    Accuracy:0.959
    Wrong count for digit 0: 10
    Wrong count for digit 1: 11
    Wrong count for digit 2: 103
    Wrong count for digit 3: 37
    Wrong count for digit 4: 57
    Wrong count for digit 5: 38
    Wrong count for digit 6: 19
    Wrong count for digit 7: 70
    Wrong count for digit 8: 37
    Wrong count for digit 9: 33


    



```python
# How many parameters are in the our model(NNClassifier) ?
# 1. print the size of weight matrix of the NNs
# 2. save the count of total number of parameters
total_params_org = 0
for index , layer in enumerate([model.fc1, model.fc2, model.fc3]):
    total_params_org += layer.weight.nelement() + layer.bias.nelement()

    print(f'Layer {index + 1} - W : {layer.weight.shape} + B: {layer.bias.shape}')

print(f'Total number of parameters: {total_params_org}')
```

    Layer 1 - W : torch.Size([1000, 784]) + B: torch.Size([1000])
    Layer 2 - W : torch.Size([2000, 1000]) + B: torch.Size([2000])
    Layer 3 - W : torch.Size([10, 2000]) + B: torch.Size([10])
    Total number of parameters: 2807010



```python
class LoRaParameterization(nn.Module):
   def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
      super().__init__()

      # random gaussian initialization for a and zero for B
      self.A = nn.Parameter(torch.zeros(rank, features_out)).to(device)
      nn.init.normal_(self.A, mean=0, std=1)
      self.B = nn.Parameter(torch.zeros(features_in, rank)).to(device)

      # scale ∆Wx by α/r , where α is a constant in r.
      # When optimizing with adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately.
      #   as a result, we simply set α to the first r we try and do not tune it.
      #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
      self.scale = alpha / rank
      self.enabled = True
   def forward(self, org_weights):
      if self.enabled:
         return org_weights + torch.matmul(self.B, self.A).view(org_weights.shape)
      else:
         return org_weights
```


```python
# addition of the parameters to our NNs
import torch.nn.utils.parametrize as parametrize

def LinearLayerParameterization(layer, device, rank=1, lora_alpha=1):
    # we only add parameterization to the weight matrix and ignore the bias
    # We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency.
    # We leave the emphirical investigation of [...] and biases to a future work

    features_in, features_out = layer.weight.shape

    return LoRAParameterization(
        features_in,
        features_out,
        rank=rank,
        alpha=lora_alpha,
        device=device
    )

parametrize.register_parametrization(
    model.fc1, 
    "weight",
    LinearLayerParameterization(model.fc1, device)
)

parametrize.register_parametrization(
    model.fc2, 
    "weight",
    LinearLayerParameterization(model.fc2, device)
)

parametrize.register_parametrization(
    model.fc3, 
    "weight",
    LinearLayerParameterization(model.fc3, device)
)

def EnableDesableLoRA(enable=True):
    for layer in [model.fc1, model.fc2, model.fc3]:
        layer.parameterization["weight"][0].enabled = enabled
```


```python
total_param_lora = 0
total_param_non_lora = 0
for index, layer in enumerate([model.fc1, model.fc2, model.fc3]):
    total_param_lora += layer.parametrizations["weight"][0].A.nelement() + layer.parametrizations["weight"][0].B.nelement()
    total_param_non_lora += layer.weight.nelement() + layer.weight.nelement()
    
    print(
       f'Layer {index+1}: W: {layer.weight.shape} + Bias: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].A.shape} + Lora_B: {layer.parametrizations["weight"][0].B.shape}'
    )


# The non-LoRA param count must match the original network

print(f'Total number of param (original): {total_param_non_lora:,}')
print(f'Total number of param (original + LoRA): {total_param_lora + total_param_non_lora:,}')
print(f'Param introduced by LoRA: {total_param_lora:,}')
param_incremment = (total_param_lora / total_param_non_lora) * 100
print(f'Param incremment: {param_incremment:.3f}%')
```

    Layer 1: W: torch.Size([1000, 784]) + Bias: torch.Size([1000]) + Lora_A: torch.Size([1, 784]) + Lora_B: torch.Size([1000, 1])
    Layer 2: W: torch.Size([2000, 1000]) + Bias: torch.Size([2000]) + Lora_A: torch.Size([1, 1000]) + Lora_B: torch.Size([2000, 1])
    Layer 3: W: torch.Size([10, 2000]) + Bias: torch.Size([10]) + Lora_A: torch.Size([1, 2000]) + Lora_B: torch.Size([10, 1])
    Total number of param (original): 5,608,000
    Total number of param (original + LoRA): 5,614,794
    Param introduced by LoRA: 6,794
    Param incremment: 0.121%



```python
# 1. Freeze all params of the original model
for name, param in model.named_parameters():
    if 'lora' not in name:
        print(f"Freezing Original params of models :  {name}")
        param.requires_grad=False

        
        
# 2. Finetuning the models params that introduced by LoRA on training digit 2
# let's load the mnist data set and keeping only the digit 2
train = datasets.MNIST(
    './dataset',
    train=True,
    download=True,
    transform=transform,
    )
exclud_idx = train.targets = 2
train.data = train.data[exclud_idx]
train.targets = train.targets[exclud_idx]

# creating the dataloader for the training
train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=10,
    suffle=True)

# Train the network with LoRA only on the digit 2 and only for 100 batches (hoping that it would improve the performance on the digit 9)
train(train_loader, model, epochs=1, total_iterations_limit=100)
```


```python
# Verify that the fine-tuning didn't alter the original weights, but only the ones introduced by LoRA.
# Check that the frozen parameters are still unchanged by the finetuning
assert torch.all(net.linear1.parametrizations.weight.original == original_weights['linear1.weight'])
assert torch.all(net.linear2.parametrizations.weight.original == original_weights['linear2.weight'])
assert torch.all(net.linear3.parametrizations.weight.original == original_weights['linear3.weight'])

enable_disable_lora(enabled=True)
# The new linear1.weight is obtained by the "forward" function of our LoRA parametrization
# The original weights have been moved to net.linear1.parametrizations.weight.original
# More info here: https://pytorch.org/tutorials/intermediate/parametrizations.html#inspecting-a-parametrized-module
assert torch.equal(net.linear1.weight, net.linear1.parametrizations.weight.original + (net.linear1.parametrizations.weight[0].lora_B @ net.linear1.parametrizations.weight[0].lora_A) * net.linear1.parametrizations.weight[0].scale)

enable_disable_lora(enabled=False)
# If we disable LoRA, the linear1.weight is the original one
assert torch.equal(net.linear1.weight, original_weights['linear1.weight'])
```


```python
#Test the network with LoRA enabled (the digit 9 should be classified better)
# Test with LoRA enabled
enable_disable_lora(enabled=True)
test()
# Test the network with LoRA disabled (the accuracy and errors counts must be the same as the original network)
# Test with LoRA disabled
enable_disable_lora(enabled=False)
test()

```

## Reference
1. https://huggingface.co/EleutherAI/gpt-neo-125m
2. https://huggingface.co/google-t5/t5-small
3. https://huggingface.co/datasets/stanfordnlp/imdb

## Tutorials
1. Problem : https://colab.research.google.com/drive/1TyF-FmHN1Yd72qdaX-Z_a6kpcvWmH1FP?usp=sharing#scrollTo=_8C1P8OnezTP
2. Solutions : https://colab.research.google.com/drive/1t0FqAqS2m3eGHsSKlQnlidgrW4IcHjCw?usp=sharing#scrollTo=X7Eb4LrVzUM0
3. Summer School : https://colab.research.google.com/drive/1OkqcpLVbze_obiomPakArA5kabWbn3lO#scrollTo=kKGxLfu0wS-U

- HandsOn Tutorial Notebook by Prof. Ashutosh Modi: https://tinyurl.com/PEFT-HandsOn-Sol
- Huggingface PEFT Library: https://github.com/huggingface/peft/tree/main
- LoRA Paper: https://arxiv.org/pdf/2106.09685
- LoRA Explanation by Umar Jamil - https://www.youtube.com/watch?v=PXWYUTMt-AU
- QLoRA Paper: https://arxiv.org/pdf/2305.14314
- OLoRA Paper Explained: https://www.youtube.com/watch?v=6l8GZDPbFn8
- PEFT Hugging Face : https://huggingface.co/blog/samuellimabraz/peft-methods
