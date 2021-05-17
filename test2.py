from mdlf import optimizer, loss
import torch
import mdlf.models as models
import mdlf.layers as layers
import mdlf.activations as activations
import mdlf.metrics as metrics

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
import dlc_practical_prologue as prologue

#to be deleted when submission
import torch
torch.manual_seed(42)




# x = None
# w1 = None
# b1 = None
# w2 = None
# b2 

######################################################################
def sigma(x):
    return x.tanh()
def dsigma(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
######################################################################
def tloss(v, t):
    return (v - t).pow(2).sum()
def dloss(v, t):
    return 2 * (v - t)
######################################################################
def forward_pass(w1, b1, w2, b2, x):
    x0 = x
    s1 = w1.mv(x0) + b1
    x1 = sigma(s1)
    s2 = w2.mv(x1) + b2
    x2 = sigma(s2)
    print(x2)
    return x0, s1, x1, s2, x2

def backward_pass(w1, b1, w2, b2,
                  t,
                  x, s1, x1, s2, x2,
                  dl_dw1, dl_db1, dl_dw2, dl_db2):
    x0 = x
    dl_dx2 = dloss(x2, t)
    # print("back")
    # print(dl_dx2)
    dl_ds2 = dsigma(s2) * dl_dx2
    # print(dl_ds2)
    dl_dx1 = w2.t().mv(dl_ds2)
    # print(dl_dx1)
    dl_ds1 = dsigma(s1) * dl_dx1
    # print(dl_ds1)
    # print("backend")

    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1)
######################################################################

train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels = True,
                                                                        normalize = True)

nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)
zeta = 0.90
train_target = train_target * zeta
test_target = test_target * zeta
nb_hidden = 8
eta = 1e-1 / nb_train_samples
epsilon = 1e-6

input_size = 5

uniform_param_1 = 1 / (5 ** (1/2)) # (train_input.size(1)**(1/2))
w1 = torch.empty(nb_hidden, 5 ).uniform_(-uniform_param_1, uniform_param_1)#train_input.size(1)
print(w1)
b1 = torch.empty(nb_hidden).fill_(0)
uniform_param_2 = 1 / (nb_hidden**(1/2))
w2 = torch.empty(nb_classes, nb_hidden).uniform_(-uniform_param_2, uniform_param_2)
# print(w2)
b2 = torch.empty(nb_classes).fill_(0)
dl_dw1 = torch.empty(w1.size())
dl_db1 = torch.empty(b1.size())
dl_dw2 = torch.empty(w2.size())
dl_db2 = torch.empty(b2.size())

print()
num_epoch = 1

for _ in range(num_epoch):
    # Back-prop
    acc_loss = 0
    nb_train_errors = 0
    
    for n in range(1):
        dl_dw1.zero_()
        dl_db1.zero_()
        dl_dw2.zero_()
        dl_db2.zero_()
        x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, train_input[n, 0 : 5])
        pred = x2.max(0)[1].item()
        print("pred1:",pred)
        print(train_target[n])
        print(x2)
        if train_target[n, pred] < 0.5: nb_train_errors = nb_train_errors + 1
        acc_loss = acc_loss + tloss(x2, train_target[n])
        backward_pass(w1, b1, w2, b2,
                      train_target[n],
                      x0, s1, x1, s2, x2,
                      dl_dw1, dl_db1, dl_dw2, dl_db2)
        # Gradient step
        w1 = w1 - eta * dl_dw1
        
        b1 = b1 - eta * dl_db1
        print("qqq")
        print(b1)
        w2 = w2 - eta * dl_dw2
        b2 = b2 - eta * dl_db2
        print("qqq")
        print(b2)
        print("qqq")

    print(' acc_train_loss {:.02f} acc_train_error {:.02f}%'
      .format(acc_loss,
              (100 * nb_train_errors) / 1))




model = models.Sequential()

model.add(layers.Linear(number_nodes=nb_hidden, input_dim=5))
model.add(activations.Tanh())
model.add(layers.Linear(number_nodes=nb_classes))
model.add(activations.Tanh())

model.compile(optimizer=optimizer.SGD(lambda_ = eta), 
              loss=loss.MSE(), 
              metrics=metrics.test_3())

print(model)

#two way to call function train -> input only train or train + test
# print(train_input.shape)
train_loss_per_epochs, train_accuracy_per_epochs = model.train(train_input[0: 1, 0 : 5], train_target[0 : 1], epochs=num_epoch)
#train_loss_per_epochs, train_accuracy_per_epochs, test_loss_per_epochs, test_accuracy_per_epochs = model.train(train_data, train_label,epochs = 10, test_data =test_data, test_label= test_label)

#fit function 
# test_loss,test_accuracy,predicted_labels = model(test_input,test_target)