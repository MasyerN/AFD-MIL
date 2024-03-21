import torch
import torch.nn as nn
import torch.nn.functional as F



class ABMIL(nn.Module):
    def __init__(self, cfgs):
        super(ABMIL, self).__init__()
        self.L = cfgs['feature_dim']
        self.D = int(cfgs['feature_dim'] / 2)
        self.K = 1

        #self.encoder = MeanPoolingMIL(cfgs)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(.2),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            # nn.Softmax(dim=-1)
            # nn.Sigmoid()
        )
    def forward(self, x):

        attn = self.attention(x).softmax(dim=1)
        m = torch.matmul(attn.transpose(-1, -2), x).squeeze(1)  # Bx1xN
        y = self.classifier(m)
        # y_hat = torch.ge(y_prob, 0.5).float()
        return y, attn

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        return y.softmax(dim=1), y.argmax(dim=-1)


class MixABMIL(nn.Module):
    def __init__(self, cfgs):
        super(MixABMIL, self).__init__()
        self.L = cfgs['feature_dim']
        self.D = int(cfgs['feature_dim'] / 2)
        self.K = 1

        self.encoder = MeanPoolingMIL(cfgs)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Dropout(.2),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            # nn.Softmax(dim=-1)
            # nn.Sigmoid()
        )
    def forward(self, x):
        y_, _ = self.encoder(x[0])
        attn = self.attention(_).softmax(dim=0)
        m = torch.matmul(attn.transpose(-1, -2), _) # Bx1xN
        y = self.classifier(m)
        # y_hat = torch.ge(y_prob, 0.5).float()
        return y, y_

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        y = y.argmax(dim=-1)
        return y
    
    def inference2(self, x):
        _, x = self.encoder(x[0])
        attn = self.attention(x.unsqueeze(0)).softmax(dim=1)
        m = torch.matmul(attn.transpose(-1, -2), x).squeeze(1)  # Bx1xN
        y = self.classifier(m)
        y = y.argmax(dim=-1)
        return y, attn.transpose(-1, -2).squeeze(0).squeeze(0)




class GatedAttention(nn.Module):
    def __init__(self, cfgs):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        self.feature_extractor_part = nn.Sequential(
            nn.Linear(1024, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, cfgs["num_classes"]),
            nn.Sigmoid()
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(-1, 1024)
        x = self.feature_extractor_part(x)  # NxL

        attn_V = self.attention_V(x)  # NxD
        attn_U = self.attention_U(x)  # NxD
        attn = self.attention_weights(attn_V * attn_U) # element wise multiplication # NxK
        attn = torch.transpose(attn, 1, 0)  # KxN
        attn = F.softmax(attn, dim=1)  # softmax over N

        m = torch.mm(attn, x)  # KxL

        y = self.classifier(m)

        return y, attn

    # AUXILIARY METHODS
    def inference(self, x):
        y, _ = self.forward(x)
        y = y.argmax(dim=-1)
        return y

class MeanPoolingMIL(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.embedding = nn.Linear(cfgs['feature_dim'], cfgs['feature_dim'])
        self.dropout = nn.Dropout(.2)
        self.ln = nn.LayerNorm(cfgs['feature_dim'])
        self.classifier = nn.Linear(cfgs['feature_dim'], cfgs["num_classes"])
        self.cfg = cfgs
        self.attentionnet = ABMIL(cfgs)
        
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x = self.ln(x)
        y = self.classifier(x)
        # y = F.sigmoid(y)
        return y, x
    
    def inference(self, x):
        atten_target, atten_score = self.attentionnet(x)
        x = x.reshape([-1, self.cfg['feature_dim']])
        y, _ = self.forward(x)
        return torch.softmax(y, 1), atten_target, atten_score[0]
