#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
from hymavi.training.loss_functions.cbr_loss import CustomLoss

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

        self.cbr = CustomLoss()
        self.loss_weights = [50,2,2,0.5]
        

    def forward(self, x, y, feat, num_classes):
        # print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy " , x[0]), type(y[0]), type(z[0])
        # print("////////////////////////////////////##############################", x[0].shape, y[1].shape, z.shape)
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        # print("++++++++++++++++++++++++++++kkkkkkkkkkkkkk ", x[0].shape, x[-1].shape)
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&& ", x[0].shape, y[0].shape, feat.shape, num_classes)
        cbr_losses = self.cbr(x[0], y[0].squeeze(1), feat, num_classes)
        losses = [l] + cbr_losses
        loss = 0
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& ", losses)
        for i in range(len(losses)):
            # print(f"///////////////////////{i} ", losses[i]*self.loss_weights[i])
            loss = loss + losses[i]*self.loss_weights[i]
        # print("/////^^^^^^^^^^^^^ ", loss)
        return loss
